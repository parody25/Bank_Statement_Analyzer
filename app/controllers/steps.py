import os 
from dotenv import load_dotenv
import pandas as pd
from typing import Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field, constr
from pprint import pprint

# ---- LlamaIndex (OpenAI backend) ----
from llama_index.llms.openai import OpenAI
#from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.core import Settings

#Langchain Imports 
from langchain.schema.runnable import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

_ = load_dotenv()  # Load environment variables from .env file
llm = OpenAI(model="o3-mini")  # change to your preferred model
Settings.llm = llm

#Credit Analysis
class ClassificationEnum(str, Enum):
    recurring = "recurring"
    irregular = "irregular"
    others = "others"

class DebitClassificationEnum(str, Enum):
    emi_loan = "emi_loan"
    utility_payment = "utility_payment"
    others = "others"

class DebitRowClassification(BaseModel):
    row_id: int = Field(..., description="The original row_id from the table")
    classification: DebitClassificationEnum
    explanation: constr(min_length=3)  # type: ignore # brief why you chose that class

class RowClassification(BaseModel):
    row_id: int = Field(..., description="The original row_id from the table")
    classification: ClassificationEnum
    explanation: constr(min_length=3)  # type: ignore # brief why you chose that class

class DebitFinalResult(BaseModel):
    #month: str = Field(..., description="YYYY-MM formatted month of the table provided")
    results: List[DebitRowClassification] = Field(
        ...,
        description="List of classifications for each row in the table",
    )

class FinalResult(BaseModel):
    #month: str = Field(..., description="YYYY-MM formatted month of the table provided")
    results: List[RowClassification] = Field(
        ...,
        description="List of classifications for each row in the table",
    )


def credit_analysis(document: pd.DataFrame) -> pd.DataFrame:  # type: ignore  # Changed from List[Dict[str, Any]] to pd.DataFrame
#List[Dict[str, Any]]:
    """
    Analyze credit transactions in the provided DataFrame.
    
    Args:
        
        document (pd.DataFrame): DataFrame containing transaction data.
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries with credit analysis results.
    """
    # Assuming 'Credit' is a column in the DataFrame
    if 'Credit' not in document.columns or 'Narrative' not in document.columns:
        raise ValueError("The DataFrame must contain a 'Credit' column for analysis.")
   
    #credit_df = document[document["Type"].str.lower() == "credit"]
    credit_df = document[document["Type"].str.lower() == "credit"].copy()

    # Step 3: reset index after filtering
    credit_df = credit_df.reset_index(drop=True)

    # Step 4: add row_id
    credit_df["row_id"] = credit_df.index

    # Select only the 5 columns
    credit_df = credit_df[["row_id","TransactionDate", "Narrative", "Credit", "Type"]].dropna()
    
    # Convert TransactionDate to datetime
    credit_df["TransactionDate"] = pd.to_datetime(credit_df["TransactionDate"])
    # Group by Year-Month automatically
    monthly_data = credit_df.groupby(credit_df["TransactionDate"].dt.to_period("M"))

    #cols 
    cols = ["row_id", "TransactionDate", "Narrative", "Credit","Type"]
    all_results: list[RowClassification] = []
    for period, month_df in monthly_data:
        print(f"\nMonth: {period}")
        print(month_df)
        # Build month-specific markdown
        # Important: include the stable row_id for mapping back
        month_label = str(period)  # e.g., "2025-08"
        md_table = month_df[cols].copy()
        # Pretty date for readability (optional)
        md_table["TransactionDate"] = md_table["TransactionDate"].dt.strftime("%Y-%m-%d")
        table_markdown = md_table.to_markdown(index=False)
        # Call the LLM for this month
        with open('app/prompts/credit_classification.txt', 'r', encoding='utf-8') as f:
            user_prompt_template = f.read() 
        prompt = PromptTemplate.from_template(user_prompt_template)
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        #structured_llm = llm.with_structured_output(StandardizeResponse)
        structured_llm = llm.with_structured_output(method="function_calling",schema=FinalResult.model_json_schema())
        #pprint(FinalResult.model_json_schema())
        chain: Runnable = prompt | structured_llm
        response = chain.invoke({
            "month": month_label,
    "table_markdown": table_markdown,
})
        #print("Model response:", response)
        print("Response type:", type(response))
        all_results.extend(response["results"])  # type: ignore

    # Convert list to DataFrame
    results_df = pd.DataFrame(all_results)
    credit_df = credit_df.merge(results_df, on='row_id', how='left') 
    print("Final DataFrame with classifications:")
    print(credit_df)

    

    result = credit_df 
    return result

def debit_analysis(document: pd.DataFrame) -> pd.DataFrame:  # type: ignore
#List[Dict[str, Any]]:
    """
    Analyze debit transactions in the provided DataFrame.
    
    Args:
        document (pd.DataFrame): DataFrame containing transaction data.
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries with debit analysis results.
    """
    # Assuming 'Debit' is a column in the DataFrame

    if 'Debit' not in document.columns or 'Narrative' not in document.columns:
        raise ValueError("The DataFrame must contain a 'Debit' column for analysis.")
    
    #credit_df = document[document["Type"].str.lower() == "credit"]
    debit_df = document[document["Type"].str.lower() == "debit"].copy()

    # Step 3: reset index after filtering
    debit_df = debit_df.reset_index(drop=True)

    # Step 4: add row_id
    debit_df["row_id"] = debit_df.index

    # Select only the 5 columns
    debit_df = debit_df[["row_id","TransactionDate", "Narrative", "Debit", "Type"]].dropna()

    # Convert TransactionDate to datetime
    debit_df["TransactionDate"] = pd.to_datetime(debit_df["TransactionDate"])
    # Group by Year-Month automatically
    monthly_data = debit_df.groupby(debit_df["TransactionDate"].dt.to_period("M"))

    #cols 
    cols = ["row_id", "TransactionDate", "Narrative", "Debit","Type"]
    all_results: list[DebitRowClassification] = []
    for period, month_df in monthly_data:
        print(f"\nMonth: {period}")
        print(month_df)
        # Build month-specific markdown
        # Important: include the stable row_id for mapping back
        month_label = str(period)  # e.g., "2025-08"
        md_table = month_df[cols].copy()
        # Pretty date for readability (optional)
        md_table["TransactionDate"] = md_table["TransactionDate"].dt.strftime("%Y-%m-%d")
        table_markdown = md_table.to_markdown(index=False)
        # Call the LLM for this month
        with open('app/prompts/debit_classification.txt', 'r', encoding='utf-8') as f:
            user_prompt_template = f.read() 
        prompt = PromptTemplate.from_template(user_prompt_template)
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        #structured_llm = llm.with_structured_output(StandardizeResponse)
        structured_llm = llm.with_structured_output(method="function_calling",schema=DebitFinalResult.model_json_schema())
        #pprint(FinalResult.model_json_schema())
        chain: Runnable = prompt | structured_llm
        response = chain.invoke({
            "month": month_label,
    "table_markdown": table_markdown,
})
        print("Model response:", response)
        print("Response type:", type(response))
        all_results.extend(response["results"])  # type: ignore

    # Convert list to DataFrame
    results_df = pd.DataFrame(all_results)
    debit_df = debit_df.merge(results_df, on='row_id', how='left') 
    print("Final DataFrame with classifications:")
    print(debit_df)



    result = debit_df
    return result


def surplus_commentry(document: pd.DataFrame) -> str:
    """
    Generate commentary on surplus transactions in the provided DataFrame.
    
    Args:
        document (pd.DataFrame): DataFrame containing transaction data.
        
    Returns:
        str: Commentary on surplus transactions.
    """
    # Assuming 'Balance' is a column in the DataFrame
    if 'Balance' not in document.columns:
        raise ValueError("The DataFrame must contain a 'Balance' column for analysis.")
    surplus_data = document[['TransactionDate', 'Balance']].dropna()
    # Calculate surplus commentary based on the balance
    surplus_commentary = "Surplus analysis not implemented yet."
    # Example logic for surplus commentary  
    if not surplus_data.empty:
        total_surplus = surplus_data['Balance'].sum()
        surplus_commentary = f"Total surplus over the period: {total_surplus:.2f}"
    else:
        surplus_commentary = "No surplus transactions found."
     
    return surplus_commentary

def dti_commentry(document: pd.DataFrame) -> str:
    """
    Generate commentary on debt-to-income ratio in the provided DataFrame.
    
    Args:
        document (pd.DataFrame): DataFrame containing transaction data.
        
    Returns:
        str: Commentary on debt-to-income ratio.
    """
    # Assuming 'Credit' and 'Debit' are columns in the DataFrame
    if 'Credit' not in document.columns or 'Debit' not in document.columns:
        raise ValueError("The DataFrame must contain 'Credit' and 'Debit' columns for analysis.")
    
    total_credit = document['Credit'].sum()
    total_debit = document['Debit'].sum()
    
    if total_credit == 0:
        return "No credit transactions found, cannot calculate DTI."
    
    dti_ratio = total_debit / total_credit
    dti_commentary = f"Debt-to-Income Ratio (DTI): {dti_ratio:.2f}"
    
    return dti_commentary



def behavioral_commentry(document: pd.DataFrame) -> str:
    """
    Generate commentary on behavioral score in the provided DataFrame.
    
    Args:
        document (pd.DataFrame): DataFrame containing transaction data.
        
    Returns:
        str: Commentary on behavioral score.
    """
    # Assuming 'Credit' and 'Debit' are columns in the DataFrame
    if 'Credit' not in document.columns or 'Debit' not in document.columns:
        raise ValueError("The DataFrame must contain 'Credit' and 'Debit' columns for analysis.")
    
    total_credit = document['Credit'].sum()
    total_debit = document['Debit'].sum()
    
    if total_credit == 0:
        return "No credit transactions found, cannot calculate behavioral score."
    
    behavioral_score = (total_credit - total_debit) / total_credit * 100
    behavioral_commentary = f"Behavioral Score: {behavioral_score:.2f}%"
    
    return behavioral_commentary

