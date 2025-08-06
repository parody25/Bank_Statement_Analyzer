import os
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
_ = load_dotenv()  # Load environment variables from .env file

class DocumentEvent(Event):
    document: str

class ClassificationEvent(Event):
    classification: str

class ReportEvent(Event):
    report: str

class ComplianceEvent(Event):
    status: str

class BankStatementAnalyzer(Workflow):
    llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

    @step
    async def document_classification(self, ev: StartEvent) -> ClassificationEvent:
        document = ev.document
        prompt = f"Classify the following document:\n\n{document}"
        #prompt = f"Classify the document content: How are you ?"
        response = await self.llm.acomplete(prompt)
        print(f"Classification response: {response}")
        return ClassificationEvent(classification=str(response))
    
    @step
    async def report_generation(self, ev: ClassificationEvent) -> ReportEvent:
        classification = ev.classification
        prompt = f"Generate a report based on the following classification:\n\n{classification}"
        response = await self.llm.acomplete(prompt)
        print(f"Report generation response: {response}")
        return ReportEvent(report=str(response))
    
    @step
    async def compliance_step(self, ev: ReportEvent) -> StopEvent:
        # Final step, 
        # You can extend this to do real compliance checks
        return StopEvent(result="Compliance check passed (mocked).")
    



# Running the workflow
#workflow_run = BankStatementAnalyzer(timeout=60, verbose=False)
#result = await workflow_run.run(document="Bank statement content goes here.")
#print(str(result))
