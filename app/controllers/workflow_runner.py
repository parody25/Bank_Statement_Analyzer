from controllers.workflow_definition import BankStatementAnalyzer
from llama_index.utils.workflow import (
    draw_all_possible_flows,
    draw_most_recent_execution,
)


# Running the workflow
async def run():
    workflow_run = BankStatementAnalyzer(timeout=300, verbose=False)
    result = await workflow_run.run(document="Synthetic Document")
    #print(str(result))
    print("Workflow run completed successfully.")
    print("Report:", result.report)
    draw_all_possible_flows(BankStatementAnalyzer, filename="bank_statement_flow_all.html")
    draw_most_recent_execution(workflow_run, filename="bank_statement_flow_recent.html")
    return result.report




    
    
