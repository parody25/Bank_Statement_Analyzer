from pydantic import BaseModel
from typing import List, Dict, Optional

class Transaction(BaseModel):
    date: str
    description: str
    amount: float
    type: str  # "credit" or "debit"
    balance: Optional[float] = None

class AnalysisRequest(BaseModel):
    bank_data: List[Transaction]
    months: int = 6

class IncomeAnalysis(BaseModel):
    salary: float
    rental: float
    other_income: float
    income_stability: str
    salary_consistency: int  # Number of consistent salary deposits

class ExpenseAnalysis(BaseModel):
    recurring: float
    utilities: float
    discretionary: float
    cash_withdrawals: float
    essential_expenses: float

class DebtAssessment(BaseModel):
    total_emis: float
    dti_ratio: float
    credit_card_payments: float
    loan_types: List[str]

class RiskIndicators(BaseModel):
    bounced_cheques: int
    overdraft_count: int
    gambling_transactions: int
    bnpl_usage: int
    high_risk_categories: List[str]

class AnalysisResponse(BaseModel):
    income: IncomeAnalysis
    expenses: ExpenseAnalysis
    debt: DebtAssessment
    risk: RiskIndicators
    surplus: float
    narrative: str
    red_flags: List[str]