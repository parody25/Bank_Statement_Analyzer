import os
import glob
import re
from typing import Optional, List

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
    """
    Find a file in DATA_DIR.
    If uuid_str is provided, look for file_type_uuid.csv.
    Otherwise return the latest file matching file_type_*.csv
    """
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
    """
    Normalize column names to known set if possible.
    Also drop the transaction reference column if present.
    """
    # Map common variants to canonical names
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
            new_cols[col] = col  # leave as-is

    df = df.rename(columns=new_cols)
    # drop TransactionReference if present
    if "TransactionReference" in df.columns:
        df = df.drop(columns=["TransactionReference"])
    return df


def build_batch_prompt(rows: List[dict]) -> str:
    """
    Build a prompt for the LLM to classify a batch of rows.
    We instruct the model to return only labels 'Recurring' or 'Irregular'
    one-per-line in the same order as the rows.
    """
    instructions = (
        "You are given a list of bank transactions. For each transaction, "
        "classify it as exactly one of these labels: Recurring or Irregular.\n\n"
        "Rules:\n"
        "- Use exactly 'Recurring' or 'Irregular' (capitalized exactly like this).\n"
        "- Output only the labels, one per line, in the same order as the input rows.\n"
        "- No additional commentary or formatting.\n\n"
        "Example:\n"
        "Monthly Salary | Debit: 0.0 | Credit: 2512.0  -> Recurring\n\n"
        "Now classify the following transactions:\n"
    )

    lines = []
    for r in rows:
        narrative = (str(r.get("Narrative", "")) or "").replace("\n", " ").strip()
        debit = r.get("Debit", 0)
        credit = r.get("Credit", 0)
        lines.append(f"Narrative: {narrative} | Debit: {debit} | Credit: {credit}")

    prompt = instructions + "\n".join(lines) + "\n\nRespond with labels now:\n"
    return prompt


def parse_labels_from_response(text: str, expected_count: int) -> List[str]:
    """
    Parse LLM response into a list of labels (Recurring/Irregular).
    If parsing fails or counts mismatch, returns an empty list to signal fallback.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    labels = []
    for ln in lines:
        # try to find recurring or irregular in the line
        if re.search(r"\brecurr", ln, re.I):
            labels.append("Recurring")
        elif re.search(r"\birreg", ln, re.I):
            labels.append("Irregular")
        # allow lines that are exactly the words
        elif ln.lower() in ("recurring", "irregular"):
            labels.append(ln.capitalize())

    if len(labels) == expected_count:
        return labels
    # fallback: if not exact match, try filtering lines with only expected words
    simple = [ln for ln in lines if ln.lower() in ("recurring", "irregular")]
    if len(simple) == expected_count:
        return [s.capitalize() for s in simple]

    # failed to parse matching count
    return []


async def classify_dataframe(df: pd.DataFrame) -> List[str]:
    """
    Classify the dataframe rows in batches. Returns list of labels same length as df.
    If LLM parsing fails for a batch, fallback to classify each row individually.
    """
    labels = []
    rows = df.to_dict(orient="records")

    # process in batches
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        prompt = build_batch_prompt(batch)
        try:
            resp = await llm.acomplete(prompt)
            text = str(resp).strip()
            parsed = parse_labels_from_response(text, expected_count=len(batch))
            if not parsed:
                # fallback to per-row classification
                parsed = []
                for r in batch:
                    single_prompt = (
                        "Classify this single transaction as exactly one of: Recurring or Irregular.\n\n"
                        f"Narrative: {r.get('Narrative','')} | Debit: {r.get('Debit',0)} | Credit: {r.get('Credit',0)}\n"
                        "Respond with exactly one word: Recurring or Irregular."
                    )
                    sresp = await llm.acomplete(single_prompt)
                    stext = str(sresp).strip()
                    if re.search(r"\brecurr", stext, re.I):
                        parsed.append("Recurring")
                    elif re.search(r"\birreg", stext, re.I):
                        parsed.append("Irregular")
                    elif stext.lower() in ("recurring", "irregular"):
                        parsed.append(stext.capitalize())
                    else:
                        parsed.append("Irregular")  # conservative default
            labels.extend(parsed)
        except Exception as e:
            # if the LLM failed, mark batch as Irregular to be conservative
            labels.extend(["Irregular"] * len(batch))

    # final safety: ensure same length
    if len(labels) != len(rows):
        # fill remaining with Irregular
        while len(labels) < len(rows):
            labels.append("Irregular")
        labels = labels[: len(rows)]
    return labels


def save_classified_csv(df: pd.DataFrame, original_path: str, kind: str) -> str:
    """
    Save classified CSV next to original with suffix _classified.
    Returns the saved path.
    """
    uuid_part = extract_uuid_from_filename(original_path)
    out_name = f"classified_{kind}_{uuid_part}.csv"
    out_path = os.path.join(DATA_DIR, out_name)
    df.to_csv(out_path, index=False)
    return out_path


async def _run_analysis_for_kind(kind: str, uuid_str: Optional[str]) -> dict:
    """
    Common runner for credit or debit.
    kind should be 'credit' or 'debit'.
    """
    path = find_file(kind, uuid_str)
    df = pd.read_csv(path)
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV is empty")

    df = normalize_df_columns(df)

    # ensure required columns exist (best-effort)
    if "Narrative" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV missing Narrative column")
    if "Debit" not in df.columns and "Credit" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV missing Debit/Credit columns")

    labels = await classify_dataframe(df)

    # Drop reference column if any leftover (redundant safety)
    ref_cols = [c for c in df.columns if "reference" in c.lower()]
    if ref_cols:
        df = df.drop(columns=ref_cols)

    # Add classification column
    df["Classification"] = labels

    saved_path = save_classified_csv(df, path, kind)
    # also return small sample to caller
    sample = df.head(10).to_dict(orient="records")
    return {"original": path, "classified_path": saved_path, "rows": len(df), "sample": sample}


@router.post("/credit")
async def analyze_credit(uuid: Optional[str] = Query(None, description="UUID part of credit_<uuid>.csv (optional)")):
    """
    Classify the credit CSV. If uuid not provided picks the latest credit_*.csv in DATA_DIR.
    """
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
    """
    Classify the debit CSV. If uuid not provided picks the latest debit_*.csv in DATA_DIR.
    """
    try:
        result = await _run_analysis_for_kind("debit", uuid)
        return JSONResponse(content=result)
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
