
from pathlib import Path
import os
import pandas as pd
from controllers.workflow_definition import BankStatementAnalyzer
from llama_index.utils.workflow import (
    draw_all_possible_flows,
    draw_most_recent_execution,
)


BASE_DIR: Path = Path(__file__).resolve().parent.parent
FINAL_CSV_DIR = f"{BASE_DIR}/uploads/final_csv_uploads"
os.makedirs(FINAL_CSV_DIR, exist_ok=True)

# Running the workflow
async def run():
    # Get the file path
    files = list(Path(FINAL_CSV_DIR).iterdir())
    # Check if any files are found
    if not files:
        raise FileNotFoundError(f"No file found in {FINAL_CSV_DIR}")
    csv_file = files[0]  # Only one file expected
    print(f"Processing file: {csv_file}")
    df = pd.read_csv(csv_file,parse_dates=['TransactionDate'])
    print(df.head())  # Preview first few rows  
    print(f"Running workflow on file: {csv_file}")
    print("Data types before conversion:", df.dtypes)
    print("Row count before conversion:", len(df))
    #Change the dataframe datatype 
    if "TransactionDate" in df.columns:
        df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce', dayfirst=True)
        print("TransactionDate converted to datetime:", df['TransactionDate'].dtypes)
        print("TransactionDate values:", df['TransactionDate'].head())
    if "Credit" in df.columns:
        df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce').astype(float).fillna(0.0)
        print("Credit converted to float:", df['Credit'].dtypes)
        print("Credit values:", df['Credit'].head())
    if "Debit" in df.columns:
        df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce').astype(float).fillna(0.0)
        print("Debit converted to float:", df['Debit'].dtypes)
        print("Debit values:", df['Debit'].head()) 
                
    if "Balance" in df.columns:
        df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce').astype(float).fillna(0.0)
        print("Balance converted to float:", df['Balance'].dtypes)
        print("Balance values:", df['Balance'].head())
    print("Data types after conversion:", df.dtypes)
                

    workflow_run = BankStatementAnalyzer(timeout=300, verbose=False)
    result = await workflow_run.run(document=df)
    #print(str(result))
    print("Workflow run completed successfully.")
    print("Report:", result.report)
    draw_all_possible_flows(BankStatementAnalyzer, filename="bank_statement_flow_all.html")
    draw_most_recent_execution(workflow_run, filename="bank_statement_flow_recent.html")
    return result.report




    
    
