import json
import os
import glob
import re
from typing import Optional, List, Tuple

import pandas as pd
from pydantic import BaseModel
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
    
class SurplusResponse(BaseModel):
    total_income: float
    total_expenses: float
    net_surplus: float
    analysis: str

class DTIResponse(BaseModel):
    total_income: float
    total_debt_payments: float
    dti_ratio: float
    analysis: str

class BehavioralResponse(BaseModel):
    behavioral_score: int
    analysis: str

@router.post("/surplus", summary="Analyze Financial Surplus", response_model=SurplusResponse)
async def analyze_surplus(uuid: str = Query(..., description="UUID part of the classified credit/debit files.")):
    try:
        credit_path = find_file("classified_credit", uuid)
        debit_path = find_file("classified_debit", uuid)

        df_credit = pd.read_csv(credit_path)
        df_debit = pd.read_csv(debit_path)

        total_income = df_credit['Credit'].sum()
        total_expenses = df_debit['Debit'].sum()
        net_surplus = total_income - total_expenses

        prompt = (
            f"Analyze the financial health based on the following data:\n"
            f"- Total Income: {total_income:.2f}\n"
            f"- Total Expenses: {total_expenses:.2f}\n"
            f"- Net Surplus/Deficit: {net_surplus:.2f}\n\n"
            f"Provide a brief, one-paragraph qualitative analysis of this cash flow situation. "
            f"Comment on whether the surplus is healthy, marginal, or if a deficit is a concern. "
            f"Return ONLY the analysis paragraph."
        )
        resp = await llm.acomplete(prompt)
        analysis_text = str(resp).strip()

        return SurplusResponse(
            total_income=total_income,
            total_expenses=total_expenses,
            net_surplus=net_surplus,
            analysis=analysis_text
        )
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dti", summary="Analyze Debt-to-Income Ratio", response_model=DTIResponse)
async def analyze_dti(uuid: str = Query(..., description="UUID part of the classified credit/debit files.")):
    try:
        credit_path = find_file("classified_credit", uuid)
        debit_path = find_file("classified_debit", uuid)

        df_credit = pd.read_csv(credit_path)
        df_debit = pd.read_csv(debit_path)

        total_income = df_credit['Credit'].sum()
        emi_df = df_debit[df_debit['Classification'] == 'EMI']
        total_debt_payments = emi_df['Debit'].sum()
        
        dti_ratio = (total_debt_payments / total_income) * 100 if total_income > 0 else 0

        prompt = (
            f"Analyze the Debt-to-Income (DTI) ratio based on the following data:\n"
            f"- Total Income: {total_income:.2f}\n"
            f"- Total Debt (EMI) Payments: {total_debt_payments:.2f}\n"
            f"- Calculated DTI Ratio: {dti_ratio:.2f}%\n\n"
            f"Provide a brief, one-paragraph analysis. A DTI below 36% is good, 36-43% is manageable, "
            f"and above 43% is a high-risk indicator. Return ONLY the analysis paragraph."
        )
        resp = await llm.acomplete(prompt)
        analysis_text = str(resp).strip()

        return DTIResponse(
            total_income=total_income,
            total_debt_payments=total_debt_payments,
            dti_ratio=dti_ratio,
            analysis=analysis_text
        )
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/behavioral", summary="Analyze Financial Behavior", response_model=BehavioralResponse)
async def analyze_behavioral(uuid: str = Query(..., description="UUID part of the classified credit/debit files.")):
    try:
        credit_path = find_file("classified_credit", uuid)
        debit_path = find_file("classified_debit", uuid)

        df_credit = pd.read_csv(credit_path)
        df_debit = pd.read_csv(debit_path)
        
        full_df = pd.concat([df_credit, df_debit], ignore_index=True)
        full_df['TransactionDate'] = pd.to_datetime(full_df['TransactionDate'])
        full_df = full_df.sort_values(by='TransactionDate')

        # To keep the prompt clean, we'll only send the relevant columns
        report_df = full_df[['TransactionDate', 'Narrative', 'Debit', 'Credit', 'Classification']]
        transactions_md = report_df.to_markdown(index=False)
        
        prompt = (
            f"Analyze the following list of transactions to identify key financial behaviors and generate a score.\n"
            f"Return a JSON object with two keys: 'score' (an integer 0-100) and 'analysis' (a summary paragraph).\n"
            f"Scoring Guide:\n"
            f"- 80-100: Excellent (Consistent savings, disciplined spending, no risk indicators).\n"
            f"- 60-79: Good (Some savings, generally responsible spending, few risk indicators).\n"
            f"- 40-59: Average (Inconsistent savings, some impulsive spending).\n"
            f"- 0-39: Concerning (Little to no savings, high-risk spending, multiple red flags).\n\n"
            f"Transactions:\n{transactions_md}\n\n"
            f"Respond with ONLY the raw JSON object."
        )
        resp = await llm.acomplete(prompt)
        
        raw_text = resp.text.strip()
        json_start = raw_text.find('{')
        json_end = raw_text.rfind('}') + 1

        if json_start == -1 or json_end == 0:
            raise HTTPException(status_code=500, detail="LLM response did not contain a valid JSON object.")

        json_string = raw_text[json_start:json_end]
        
        try:
            result_json = json.loads(json_string)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail=f"Failed to parse cleaned JSON from LLM response: {json_string}")

        return BehavioralResponse(
            behavioral_score=result_json.get('score', 0),
            analysis=result_json.get('analysis', 'Analysis could not be generated.')
        )
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/report", summary="Generate Final Report")
async def generate_final_report(uuid: str = Query(..., description="UUID part of the classified credit/debit files.")):
    try:
        # --- 1. Gather all necessary data by reading classified files ---
        credit_path = find_file("classified_credit", uuid)
        debit_path = find_file("classified_debit", uuid)
        df_credit = pd.read_csv(credit_path)
        df_debit = pd.read_csv(debit_path)
        df_full = pd.concat([df_credit, df_debit], ignore_index=True)

        # --- 2. Perform calculations for all sections ---
        total_income = df_credit['Credit'].sum()
        total_expenses = df_debit['Debit'].sum()
        net_surplus = total_income - total_expenses
        
        emi_df = df_debit[df_debit['Classification'] == 'EMI']
        total_debt_payments = emi_df['Debit'].sum()
        dti_ratio = (total_debt_payments / total_income) * 100 if total_income > 0 else 0

        # --- 3. Create a comprehensive context string for the final prompt ---
        context = (
            f"## Income Analysis\n"
            f"Total Income: {total_income:.2f}\n"
            f"Income Transactions:\n{df_credit[['TransactionDate', 'Narrative', 'Credit', 'Classification']].to_markdown(index=False)}\n\n"
            f"## Expense Analysis\n"
            f"Total Expenses: {total_expenses:.2f}\n"
            f"Expense Transactions:\n{df_debit[['TransactionDate', 'Narrative', 'Debit', 'Classification']].to_markdown(index=False)}\n\n"
            f"## Cash Flow Summary\n"
            f"Net Surplus: {net_surplus:.2f}\n\n"
            f"## Debt-to-Income Summary\n"
            f"DTI Ratio: {dti_ratio:.2f}%\n\n"
        )

        # --- 4. Build the final prompt ---
        prompt = (
            f"You are a financial analyst. Synthesize the provided data into a comprehensive report. "
            f"The output must be a single, well-structured JSON object with the keys 'executive_summary', "
            f"'income_analysis', 'expense_analysis', 'cash_flow_analysis', 'debt_to_income_analysis', and "
            f"'financial_behavior_analysis'.\n\n"
            f"Use the following data to construct your report:\n{context}\n\n"
            f"Based on all the data, also generate a behavioral analysis with a score and qualitative summary. "
            f"Finally, create a 2-3 sentence executive summary for the very top of the report.\n\n"
            f"Respond with ONLY the raw JSON object."
        )

        # --- 5. Get and return the final report ---
        resp = await llm.acomplete(prompt)
        
        raw_text = resp.text.strip()
        json_start = raw_text.find('{')
        json_end = raw_text.rfind('}') + 1

        if json_start == -1 or json_end == 0:
            raise HTTPException(status_code=500, detail="LLM response did not contain a valid JSON object for the final report.")

        json_string = raw_text[json_start:json_end]
        
        try:
            report_json = json.loads(json_string)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail=f"Failed to parse cleaned JSON from LLM response for the final report: {json_string}")

        return JSONResponse(content=report_json)
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))