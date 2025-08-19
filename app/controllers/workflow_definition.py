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
    report: str


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
        response = steps.surplus_commentry(credit_classify, debit_classify)
        self.emit("surplus_analysis", {"result": str(response)})
        print("Surplus Analysis Result:", response)
        await ctx.store.set("surplus_analysis_result", response)
        return SurplusAnalysisEvent(result=response)

    @step
    async def dti_analysis(self, ctx: Context, ev: TriggerDTI) -> DebtToIncomeEvent:
        self.emit("info", {"message": "Starting step: dti_analysis"})
        credit_classify = await ctx.store.get("credit_classify_result")
        debit_classify = await ctx.store.get("debit_classify_result")
        response = steps.dti_commentry(credit_classify, debit_classify)
        self.emit("dti_analysis", {"result": str(response)})
        await ctx.store.set("dti_analysis_result", response)
        return DebtToIncomeEvent(result=response)

    @step
    async def behavioral_analysis(self, ctx: Context, ev: TriggerBehavioral) -> BehavioralScoreEvent:
        self.emit("info", {"message": "Starting step: behavioral_analysis"})
        credit_classify = await ctx.store.get("credit_classify_result")
        debit_classify = await ctx.store.get("debit_classify_result")
        response = steps.behavioral_commentry(credit_classify, debit_classify)
        self.emit("behavioral_analysis", {"result": str(response)})
        await ctx.store.set("behavioral_analysis_result", response)
        return BehavioralScoreEvent(result=response)

    @step
    async def report_generation(self, ctx: Context, ev: SurplusAnalysisEvent | DebtToIncomeEvent | BehavioralScoreEvent) -> MyStopEvent | None:
        self.emit("info", {"message": "Starting step: report_generation"})
        data = ctx.collect_events(ev, [SurplusAnalysisEvent, DebtToIncomeEvent, BehavioralScoreEvent])
        if data is None:
            return None

        surplus_event, debt_to_income_event, behavioral_score_event = data
        self.emit("pre_report", {
            "surplus": str(surplus_event.result),
            "debt_to_income": str(debt_to_income_event.result),
            "behavioral": str(behavioral_score_event.result)
        })

        with open('app/prompts/report_generation.txt', 'r', encoding='utf-8') as f:
            user_prompt_template = f.read()
        prompt = PromptTemplate.from_template(user_prompt_template)
        llm = ChatOpenAI(
            model="o3-mini",
            #temperature=0.2,
            reasoning_effort="medium",
        )
        chain = prompt | llm
        response = await chain.ainvoke({
            "surplus_analysis": surplus_event.result["result"],
            "dti_analysis": debt_to_income_event.result["result"],
            "behavior_analysis": behavioral_score_event.result["result"]
        })
        response = response.content
        self.emit("report_generated", {"report": response})
        print("Generated Report:", response)
        return MyStopEvent(report=response)
