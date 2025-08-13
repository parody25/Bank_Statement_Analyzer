import os
import glob
import re
from typing import Optional, List, Tuple

import pandas as pd
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI

load_dotenv()
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

router = APIRouter(prefix="/new_analyzer", tags=["New Analyzer"])

DATA_DIR = "app/uploads/final_csv_uploads"
BATCH_SIZE = 40  # number of rows per LLM call (adjust as needed)


def find_file(file_type: str, uuid_str: Optional[str] = None) -> str:
    if uuid_str:
        path = os.path.join(DATA_DIR, f"{file_type}_{uuid_str}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{file_type} file not found for uuid {uuid_str}")
        return path

    pattern = os.path.join(DATA_DIR, f"{file_type}_*.csv")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")
    return files[-1]


def extract_uuid_from_filename(path: str) -> str:
    name = os.path.basename(path)
    m = re.match(r"^[a-zA-Z]+_(.+?)\.csv$", name)
    return m.group(1) if m else name.rsplit(".", 1)[0]


def normalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for col in df.columns:
        lc = col.strip().lower()
        if "transaction" in lc and "date" in lc:
            new_cols[col] = "TransactionDate"
        elif lc in ("payment date", "paymentdate"):
            new_cols[col] = "PaymentDate"
        elif "narrative" in lc or "description" in lc:
            new_cols[col] = "Narrative"
        elif "reference" in lc:
            new_cols[col] = "TransactionReference"
        elif lc == "debit":
            new_cols[col] = "Debit"
        elif lc == "credit":
            new_cols[col] = "Credit"
        else:
            new_cols[col] = col
    df = df.rename(columns=new_cols)
    if "TransactionReference" in df.columns:
        df = df.drop(columns=["TransactionReference"])
    return df


def build_batch_prompt(rows: List[dict], kind: str) -> str:
    if kind == "credit":
        labels = "Recurring, Irregular, Others"
        label_rules = (
            "- 'Recurring': Regular salary, interest, etc.\n"
            "- 'Irregular': One-off or unusual income.\n"
            "- 'Others': Cash deposit.\n"
        )
    else:  # debit
        labels = "EMI, Utility Payment, Others"
        label_rules = (
            "- 'EMI': Loan EMI payments.\n"
            "- 'Utility Payment': Grocery, transport, movie, or other service payments.\n"
            "- 'Others': Cash withdrawal from account.\n"
        )

    instructions = (
        f"You are given a list of bank transactions ({kind} side). For each transaction:\n"
        f"1. Classify it as exactly one of: {labels}.\n"
        f"2. Provide a brief explanation (3–6 words) describing the transaction.\n\n"
        f"Rules:\n{label_rules}"
        "- Explanation should be short and clear.\n"
        "- Output format: <Label> | <Explanation>\n"
        "- One result per line, same order as input rows.\n\n"
        "Example:\n"
        "Recurring | Monthly salary credited\n"
        "Others | Cash deposit by self\n\n"
        "Now process the following transactions:\n"
    )

    lines = []
    for r in rows:
        narrative = (str(r.get("Narrative", "")) or "").replace("\n", " ").strip()
        debit = r.get("Debit", 0)
        credit = r.get("Credit", 0)
        lines.append(f"Narrative: {narrative} | Debit: {debit} | Credit: {credit}")

    return instructions + "\n".join(lines) + "\n\nRespond now:\n"


def parse_labels_and_explanations(text: str, expected_count: int) -> Tuple[List[str], List[str]]:
    labels, explanations = [], []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        parts = [p.strip() for p in ln.split("|", 1)]
        if len(parts) == 2:
            label, expl = parts
        else:
            label, expl = parts[0], ""
        labels.append(label)
        explanations.append(expl)
    if len(labels) != expected_count:
        return [], []
    return labels, explanations


async def classify_dataframe(df: pd.DataFrame, kind: str) -> Tuple[List[str], List[str]]:
    labels_all, expl_all = [], []
    rows = df.to_dict(orient="records")

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        prompt = build_batch_prompt(batch, kind)
        try:
            resp = await llm.acomplete(prompt)
            text = str(resp).strip()
            labels, expls = parse_labels_and_explanations(text, len(batch))

            if not labels:
                if kind == "credit":
                    labels = ["Irregular"] * len(batch)
                else:
                    labels = ["Utility Payment"] * len(batch)
                expls = ["No clear info"] * len(batch)

            labels_all.extend(labels)
            expl_all.extend(expls)
        except Exception:
            if kind == "credit":
                labels_all.extend(["Irregular"] * len(batch))
            else:
                labels_all.extend(["Utility Payment"] * len(batch)
                )
            expl_all.extend(["No clear info"] * len(batch))

    return labels_all, expl_all


def save_classified_csv(df: pd.DataFrame, original_path: str, kind: str) -> str:
    uuid_part = extract_uuid_from_filename(original_path)
    out_name = f"classified_{kind}_{uuid_part}.csv"
    out_path = os.path.join(DATA_DIR, out_name)
    df.to_csv(out_path, index=False)
    return out_path


async def _run_analysis_for_kind(kind: str, uuid_str: Optional[str]) -> dict:
    path = find_file(kind, uuid_str)
    df = pd.read_csv(path)
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV is empty")

    df = normalize_df_columns(df)

    if "Narrative" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV missing Narrative column")
    if "Debit" not in df.columns and "Credit" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV missing Debit/Credit columns")

    labels, explanations = await classify_dataframe(df, kind)
    df["Classification"] = labels
    df["Explanation"] = explanations

    saved_path = save_classified_csv(df, path, kind)
    sample = df.head(10).to_dict(orient="records")
    return {"original": path, "classified_path": saved_path, "rows": len(df), "sample": sample}


@router.post("/credit")
async def analyze_credit(uuid: Optional[str] = Query(None, description="UUID part of credit_<uuid>.csv (optional)")):
    try:
        result = await _run_analysis_for_kind("credit", uuid)
        return JSONResponse(content=result)
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/debit")
async def analyze_debit(uuid: Optional[str] = Query(None, description="UUID part of debit_<uuid>.csv (optional)")):
    try:
        result = await _run_analysis_for_kind("debit", uuid)
        return JSONResponse(content=result)
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
