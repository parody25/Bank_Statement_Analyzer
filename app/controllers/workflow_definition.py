from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context
)
import pandas as pd
from typing import Dict, Any
from . import steps
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import asyncio


class MyCustomStartEvent(StartEvent):
    document: pd.DataFrame


class TriggerCredit(Event): pass
class TriggerDebit(Event): pass
class TriggerSurplus(Event): pass
class TriggerDTI(Event): pass
class TriggerBehavioral(Event): pass


class CreditClassifyEvent(Event):
    result: pd.DataFrame


class DebitClassifyEvent(Event):
    result: pd.DataFrame


class SurplusAnalysisEvent(Event):
    result: Dict[str, Any]


class DebtToIncomeEvent(Event):
    result: Dict[str, Any]


class BehavioralScoreEvent(Event):
    result: Dict[str, Any]


class MyStopEvent(StopEvent):
    report: Dict[str, Any]
    
class BankStatementAnalyzer(Workflow):

    def emit(self, event_name, payload):
        # _emitter is set by runner; it's an async function
        if hasattr(self, "_emitter") and self._emitter:
            asyncio.create_task(self._emitter(event_name, payload))

    @step
    async def start(self, ctx: Context, ev: MyCustomStartEvent) -> TriggerCredit | TriggerDebit | None:
        self.emit("info", {"message": "Starting step: start"})
        await ctx.store.set("document", ev.document)
        self.emit("start", {"message": "Starting workflow. Document loaded."})
        ctx.send_event(TriggerCredit())
        ctx.send_event(TriggerDebit())

    @step
    async def credit_classify(self, ctx: Context, ev: TriggerCredit) -> CreditClassifyEvent:
        self.emit("info", {"message": "Starting step: credit_classify"})
        document = await ctx.store.get("document")
        response = steps.credit_analysis(document)
        await ctx.store.set("credit_classify_result", response)
        self.emit("credit_classification", {"result": str(response)})
        print("Credit Classification Result:", response)
        return CreditClassifyEvent(result=response)

    @step
    async def debit_classify(self, ctx: Context, ev: TriggerDebit) -> DebitClassifyEvent:
        self.emit("info", {"message": "Starting step: debit_classify"})
        document = await ctx.store.get("document")
        response = steps.debit_analysis(document)
        await ctx.store.set("debit_classify_result", response)
        self.emit("debit_classification", {"result": str(response)})
        print("Debit Classification Result:", response)
        return DebitClassifyEvent(result=response)

    @step
    async def join_for_surplus(self, ctx: Context, ev: CreditClassifyEvent | DebitClassifyEvent) -> TriggerSurplus | TriggerDTI | TriggerBehavioral | None:
        self.emit("info", {"message": "Starting step: join_for_surplus"})
        results = ctx.collect_events(ev, [CreditClassifyEvent, DebitClassifyEvent])
        if results is None:
            return None
        credit_event, debit_event = results
        self.emit("joined_classification", {"message": "Credit and Debit classifications completed."})
        ctx.send_event(TriggerSurplus())
        ctx.send_event(TriggerDTI())
        ctx.send_event(TriggerBehavioral())

    @step
    async def surplus_analysis(self, ctx: Context, ev: TriggerSurplus) -> SurplusAnalysisEvent:
        self.emit("info", {"message": "Starting step: surplus_analysis"})
        credit_classify = await ctx.store.get("credit_classify_result")
        debit_classify = await ctx.store.get("debit_classify_result")
        
        #Using Dataframe operation for calculation and use the LLM to Reason over it 
        # For example, calculating surplus from credit and debit classifications
        response_content = steps.surplus_commentry(credit_classify, debit_classify)
        
        # This step now generates a structured dictionary instead of just a string.
        # NEW: Structured response
        response = {
            "title": "Surplus Position",
            "content": response_content["result"] # Assuming surplus_commentry returns a dict with a 'result' key
        }
        self.emit("surplus_analysis", {"result": response.content})
        
        await ctx.store.set("surplus_analysis_result", response)
        return SurplusAnalysisEvent(result=response)

    @step
    async def dti_analysis(self, ctx: Context, ev: TriggerDTI) -> DebtToIncomeEvent:
        self.emit("info", {"message": "Starting step: dti_analysis"})
        credit_classify = await ctx.store.get("credit_classify_result")
        debit_classify = await ctx.store.get("debit_classify_result")
        
        response_content = steps.dti_commentry(credit_classify, debit_classify)
        
        # NEW: Structured response
        response = {
            "title": "Debt-to-Income (DTI) Ratio",
            "content": response_content["result"]
        }
        self.emit("dti_analysis", {"result": response.content})
        
        await ctx.store.set("dti_analysis_result", response)
        return DebtToIncomeEvent(result=response)

    @step
    async def behavioral_analysis(self, ctx: Context, ev: TriggerBehavioral) -> BehavioralScoreEvent:
        self.emit("info", {"message": "Starting step: behavioral_analysis"})
        credit_classify = await ctx.store.get("credit_classify_result")
        debit_classify = await ctx.store.get("debit_classify_result")
        
        response_content = steps.behavioral_commentry(credit_classify, debit_classify)
        
        # NEW: Structured response
        response = {
            "title": "Behavioral Insights",
            "content": response_content["result"]
        }
        self.emit("behavioral_analysis", {"result": response.content})
        await ctx.store.set("behavioral_analysis_result", response)
        return BehavioralScoreEvent(result=response)

    @step
    async def report_generation(self, ctx: Context, ev: SurplusAnalysisEvent | DebtToIncomeEvent | BehavioralScoreEvent) -> MyStopEvent | None:
        self.emit("info", {"message": "Starting step: report_generation"})
        data = ctx.collect_events(ev, [SurplusAnalysisEvent, DebtToIncomeEvent, BehavioralScoreEvent])
        
        if data is None:
            return None

        surplus_event, dti_event, behavioral_event = data
        
        # --- MAJOR CHANGE: NO MORE LLM CALL HERE ---
        # The new role of this step is to assemble the final structured report data.
        
        # You can add other sections generated by other LLM calls if needed,
        # but here we assemble the results from previous steps.
        report_data = {
            "main_title": "Standard Bank Report",
            "sections": [
                # Example of adding a manually written or LLM-generated summary
                {
                    "title": "Customer Summary",
                    "content": "The customer demonstrates several positive financial traits... (This could come from another analysis step or be generated here)"
                },
                surplus_event.result,
                dti_event.result,
                behavioral_event.result,
                # You can add more static or dynamic sections here
                {
                    "title": "Risk Assessment",
                    "content": "Credit Risk: While the overall DTI and behavioral indicators are strong... (and so on)"
                },
                {
                    "title": "Conclusion & Recommendation",
                    "content": "Final Remarks:\n The customer’s financial profile is marked by low overall debt..."
                }
            ],
            "footer": {
                "prepared_by": "Financial Analysis Department\nStandard Bank",
                "date": "[Insert Date]"
            }
        }
        
        # The final event now carries the structured dictionary
        return MyStopEvent(report=report_data)
    
    

    



# Running the workflow
#workflow_run = BankStatementAnalyzer(timeout=60, verbose=False)
#result = await workflow_run.run(document="Bank statement content goes here.")
#print(str(result))
