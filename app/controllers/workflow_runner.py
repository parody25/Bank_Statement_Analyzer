# from controllers.workflow_definition import BankStatementAnalyzer

# # Running the workflow
# async def run():
#     workflow_run = BankStatementAnalyzer(timeout=60, verbose=False)
#     result = await workflow_run.run(document="Bank statement Transactions: \n1. Deposit: $1000\n2. Withdrawal: $200\n3. Balance: $800")
#     print(str(result))
#     return str(result)

from controllers.workflow_definition import BankStatementAnalyzer, BankStatementStartEvent

async def run(document_text: str, credit_csv_path: str, debit_csv_path: str):
    workflow = BankStatementAnalyzer(timeout=300, verbose=True)
    start_ev = BankStatementStartEvent(
        document=document_text,
        credit_csv_path=credit_csv_path,
        debit_csv_path=debit_csv_path,
    )
    result = await workflow.run(start_ev)
    print(result)
    return result
