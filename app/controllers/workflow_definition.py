import os
import pandas as pd
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

class BankStatementStartEvent(StartEvent):
    credit_csv_path: str
    debit_csv_path: str
    document: str

class CreditAnalysisEvent(Event):
    classified_data: str

class DebitAnalysisEvent(Event):
    classified_data: str

class FinalAnalysisEvent(StopEvent):
    classification: str
    report: str
    credit_analysis: str
    debit_analysis: str

class AggregatedEvent(Event):
    classification_event: ClassificationEvent
    report_event: ReportEvent
    credit_event: CreditAnalysisEvent
    debit_event: DebitAnalysisEvent


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
    
    # @step
    # async def compliance_step(self, ev: ReportEvent) -> StopEvent:
    #     # Final step, 
    #     # You can extend this to do real compliance checks
    #     return StopEvent(result="Compliance check passed (mocked).")

    @step
    async def credit_analysis(self, ev: BankStatementStartEvent) -> CreditAnalysisEvent:
        csv_path = ev.credit_csv_path
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Credit CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)

        classified_rows = []
        for _, row in df.iterrows():
            prompt = (
                "Classify this transaction as 'Recurring' or 'Irregular':\n"
                f"Date: {row['TransactionDate']}\n"
                f"Narrative: {row['Narrative']}\n"
                f"Debit: {row.get('Debit', 0)}\n"
                f"Credit: {row.get('Credit', 0)}\n"
                "Answer with one word: Recurring or Irregular."
            )
            classification = await self.llm.acomplete(prompt)
            classified_rows.append({
                "TransactionDate": row['TransactionDate'],
                "PaymentDate": row.get('Payment Date', ""),
                "Narrative": row['Narrative'],
                "Debit": row.get('Debit', 0),
                "Credit": row.get('Credit', 0),
                "Classification": classification.strip().capitalize()
            })

        classified_df = pd.DataFrame(classified_rows)
        csv_classified = classified_df.to_csv(index=False)
        return CreditAnalysisEvent(classified_data=csv_classified)

    @step
    async def debit_analysis(self, ev: BankStatementStartEvent) -> DebitAnalysisEvent:
        csv_path = ev.debit_csv_path
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Debit CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)

        classified_rows = []
        for _, row in df.iterrows():
            prompt = (
                "Classify this transaction as 'Recurring' or 'Irregular':\n"
                f"Date: {row['TransactionDate']}\n"
                f"Narrative: {row['Narrative']}\n"
                f"Debit: {row.get('Debit', 0)}\n"
                f"Credit: {row.get('Credit', 0)}\n"
                "Answer with one word: Recurring or Irregular."
            )
            classification = await self.llm.acomplete(prompt)
            classified_rows.append({
                "TransactionDate": row['TransactionDate'],
                "PaymentDate": row.get('Payment Date', ""),
                "Narrative": row['Narrative'],
                "Debit": row.get('Debit', 0),
                "Credit": row.get('Credit', 0),
                "Classification": classification.strip().capitalize()
            })

        classified_df = pd.DataFrame(classified_rows)
        csv_classified = classified_df.to_csv(index=False)
        return DebitAnalysisEvent(classified_data=csv_classified)

    @step
    async def final_step(self, ev: AggregatedEvent) -> FinalAnalysisEvent:
        return FinalAnalysisEvent(
            classification=ev.classification_event.classification,
            report=ev.report_event.report,
            credit_analysis=ev.credit_event.classified_data,
            debit_analysis=ev.debit_event.classified_data,
            result="Completed all analyses"
        )

   



# Running the workflow
#workflow_run = BankStatementAnalyzer(timeout=60, verbose=False)
#result = await workflow_run.run(document="Bank statement content goes here.")
#print(str(result))


