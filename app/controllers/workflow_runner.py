from controllers.workflow_definition import BankStatementAnalyzer

# Running the workflow
async def run():
    workflow_run = BankStatementAnalyzer(timeout=60, verbose=False)
    result = await workflow_run.run(document="Bank statement Transactions: \n1. Deposit: $1000\n2. Withdrawal: $200\n3. Balance: $800")
    print(str(result))
    return str(result)