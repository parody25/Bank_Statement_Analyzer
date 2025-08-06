import json
from typing import List, Dict
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel
import logging
import os
from dotenv import load_dotenv
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# Initialize the LLM
llm = OpenAI(
    model="gpt-4o",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY")
)

def extract_json_from_response(response_text: str) -> dict:
    """Extract JSON from response text that may contain markdown code blocks"""
    try:
        # Try direct parse first
        return json.loads(response_text)
    except json.JSONDecodeError:
        # If fails, try removing markdown code blocks
        cleaned = re.sub(r'```json|```', '', response_text).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON after cleaning: {e}")
            logger.error(f"Original response: {response_text}")
            return {}

def log_raw_response(response_type: str, response: str):
    """Helper function to log raw responses"""
    logger.info(f"\nRAW {response_type.upper()} RESPONSE:\n{response}\n")

def extract_transactions(text: str) -> List[Dict]:
    """Extract transactions using LLM"""
    prompt = f"""
    Extract transactions from this bank statement text:
    {text}

    Return ONLY JSON format without any markdown code blocks:
    {{
      "transactions": [
        {{
          "date": "DD-MM-YYYY",
          "description": "Cleaned description",
          "amount": 123.45,
          "type": "debit/credit",
          "balance": 12345.67  # if available
        }}
      ]
    }}
    """
    
    response = llm.complete(prompt)
    log_raw_response("TRANSACTION EXTRACTION", response.text)
    
    try:
        data = extract_json_from_response(response.text)
        return data.get("transactions", [])
    except Exception as e:
        logger.error(f"Failed to parse transaction extraction: {e}")
        return []

def categorize_transactions(transactions: List[Dict]) -> List[Dict]:
    """Categorize transactions using LLM"""
    if not transactions:
        return []
        
    prompt = f"""
    Categorize these transactions. Be specific and don't return empty results.
    If unsure, use 'Other' category:

    Transactions:
    {json.dumps(transactions, indent=2)}

    Categories:
    - Salary income
    - Rental income
    - Other income
    - EMI payments
    - Utility payments
    - Credit card payments
    - Groceries
    - Discretionary spending (dining, entertainment, shopping)
    - Cash withdrawals
    - Subscriptions
    - Medical expenses
    - Insurance
    - Transportation
    - Education
    - Other

    Return STRICT JSON format without any markdown code blocks:
    {{
      "categorized_transactions": [
        {{
          "original_description": "Original text",
          "category": "Assigned category",
          "amount": 123.45,
          "date": "DD-MM-YYYY",
          "type": "debit/credit"
        }}
      ]
    }}
    """
    
    response = llm.complete(prompt)
    log_raw_response("CATEGORIZATION", response.text)
    
    try:
        data = extract_json_from_response(response.text)
        return data.get("categorized_transactions", [])
    except Exception as e:
        logger.error(f"Failed to parse categorization: {e}")
        return []

def analyze_finances(categorized_data: List[Dict]) -> Dict:
    """Perform financial analysis using LLM"""
    if not categorized_data:
        return {
            "income": {"salary": 0, "rental": 0, "other_income": 0, "income_stability": "Low"},
            "expenses": {"recurring": 0, "utilities": 0, "discretionary": 0, "cash_withdrawals": 0},
            "debt": {"total_emis": 0, "dti_ratio": 0.0, "credit_card_payments": 0},
            "risk": {"bounced_cheques": 0, "overdraft_count": 0},
            "surplus": 0
        }
    
    prompt = f"""
    Analyze these categorized transactions:
    {json.dumps(categorized_data, indent=2)}

    Return analysis in STRICT JSON format without any markdown code blocks:
    {{
      "income": {{
        "salary": 0,
        "rental": 0,
        "other_income": 0,
        "income_stability": "High/Medium/Low",
        "salary_consistency": 0
      }},
      "expenses": {{
        "recurring": 0,
        "utilities": 0,
        "discretionary": 0,
        "cash_withdrawals": 0,
        "essential_expenses": 0
      }},
      "debt": {{
        "total_emis": 0,
        "dti_ratio": 0.0,
        "credit_card_payments": 0,
        "loan_types": []
      }},
      "risk": {{
        "bounced_cheques": 0,
        "overdraft_count": 0,
        "gambling_transactions": 0,
        "bnpl_usage": 0,
        "high_risk_categories": []
      }},
      "surplus": 0
    }}
    """
    
    response = llm.complete(prompt)
    log_raw_response("FINANCIAL ANALYSIS", response.text)
    
    try:
        return extract_json_from_response(response.text)
    except Exception as e:
        logger.error(f"Failed to parse financial analysis: {e}")
        return {}

def generate_narrative(analysis: Dict) -> str:
    """Generate credit decision narrative"""
    prompt = f"""
    Generate mortgage underwriting narrative using this analysis:
    {json.dumps(analysis, indent=2)}

    Key points to include:
    1. Specific numbers from the analysis (don't use placeholders)
    2. Real assessment based on the actual data
    3. Clear recommendations
    4. Avoid generic templates if data is missing

    Write in formal business language for credit committee.
    """
    
    response = llm.complete(prompt)
    log_raw_response("NARRATIVE", response.text)
    return response.text

def detect_red_flags(transactions: List[Dict]) -> List[str]:
    """Detect potential red flags"""
    red_flags = []
    
    # Simple heuristic checks
    overdraft_count = sum(1 for t in transactions if "overdraft" in t.get("description", "").lower())
    if overdraft_count > 1:
        red_flags.append(f"{overdraft_count} overdraft occurrences")
    
    gambling_keywords = ["gambling", "casino", "bet", "poker"]
    if any(kw in t.get("description", "").lower() for kw in gambling_keywords for t in transactions):
        red_flags.append("Gambling transactions detected")
    
    large_cash_withdrawals = sum(t.get("amount", 0) for t in transactions 
                               if "cash withdrawal" in t.get("description", "").lower() and t.get("amount", 0) > 1000)
    if large_cash_withdrawals > 5000:
        red_flags.append(f"Large cash withdrawals: AED {large_cash_withdrawals}")
    
    return red_flags