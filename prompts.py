TRANSACTION_EXTRACTION_PROMPT = """
Extract transactions from this bank statement text. Different banks use different formats:

Format 1 (ADIB):
| Transaction Date | Payment Date    | Narrative    | Transaction Reference | Debit    | Credit    |

Format 2 (FAB):
| Date    | Value Date | Description    | Debit    | Credit    | Balance    |

Format 3 (CBD):
| Date | Description | Value Date | Debit | Credit | Balance |

Instructions:
1. Extract all transactions
2. Convert all amounts to positive numbers
3. For type: if amount is in debit column -> "debit", credit column -> "credit"
4. Use DD-MM-YYYY format for dates
5. Ignore header/footer text
6. Handle multi-page statements

Raw text:
{text}

Return ONLY JSON format:
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

CATEGORIZATION_PROMPT = """
Categorize these transactions into banking categories:

Categories:
- Salary income
- Rental income
- Other income
- EMI payments (loan repayments)
- Utility payments (electricity, water, etc.)
- Credit card payments
- Groceries
- Discretionary spending (dining, movies, shopping)
- Cash withdrawals
- Subscriptions
- Medical expenses
- Insurance
- Transportation
- Education
- Taxes/charges
- Other

Transactions:
{transactions}

Return ONLY JSON format:
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

ANALYSIS_PROMPT = """
Analyze categorized transactions for mortgage underwriting:

Key analysis areas:
1. Income Analysis: 
   - Calculate average monthly salary
   - Identify rental income
   - Assess stability (salary consistency, irregular deposits)
2. Expense Analysis:
   - Essential expenses (utilities, groceries, insurance)
   - Discretionary spending (entertainment, shopping)
   - Cash withdrawal patterns
3. Debt Assessment:
   - Total EMIs (loan repayments)
   - Credit card payments
   - Calculate DTI: (Total Debt Payments / Total Income)
4. Risk Indicators:
   - Overdraft occurrences
   - High-risk transactions (gambling, crypto)
   - BNPL usage
   - Bounced cheques
5. Surplus: (Total Income - Essential Expenses - Debt Payments)

Categorized transactions:
{categorized_transactions}

Return ONLY JSON matching this schema:
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

NARRATIVE_PROMPT = """
Generate mortgage underwriting narrative using this analysis:

{analysis_results}

Include:
1. Income stability assessment
2. Debt burden analysis (DTI ratio)
3. Risk indicators
4. Surplus cash flow
5. Red flags for fraud or financial stress
6. Overall repayment capacity assessment

Structure:
- Executive Summary
- Income Analysis
- Expense Analysis
- Debt Assessment
- Risk Factors
- Recommendation

Write in formal business language for credit committee.
Include specific numbers and percentages where available.
"""