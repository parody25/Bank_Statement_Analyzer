from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context
)
from llama_index.llms.openai import OpenAI

from llama_index.utils.workflow import (
    draw_all_possible_flows,
    draw_most_recent_execution,
)


class MyCustomStartEvent(StartEvent):
    #document: dataframe
    document: str
    
    
class TriggerCredit(Event): pass
class TriggerDebit(Event): pass

class TriggerSurplus(Event): pass
class TriggerDTI(Event): pass 
class TriggerBehavioral(Event): pass
    
class CreditClassifyEvent(Event):
    #result: 
    result: str
    
class DebitClassifyEvent(Event):
    #result:
    result: str
    
class SurplusAnalysisEvent(Event):
    #result:
    result: str
    
class DebtToIncomeEvent(Event):
    #result:
    result: str 
    
class BehavioralScoreEvent(Event):
    #result:
    result: str
    
    

class MyStopEvent(StopEvent):
    #Report: CompletionResponse
    report: str

class BankStatementAnalyzer(Workflow):
    
    
    @step
    async def start(self, ctx: Context, ev: MyCustomStartEvent) -> TriggerCredit | TriggerDebit | None:
        #Store the Dataframe in the Context
        await ctx.store.set("document", ev.document)
        
        # Trigger parallel classification steps
        ctx.send_event(TriggerCredit())  # no Payload Needed
        ctx.send_event(TriggerDebit())
        
        
    
        
    
    @step
    async def credit_classify(self, ctx: Context, ev: TriggerCredit) -> CreditClassifyEvent:
        
        #Get the Dataframe in the Context
        document = await ctx.store.get("document")
        #Filter the credit transaction using the Dataframe and Use LLM to classify the Event 
        #Store the Dataframe for credit transaction classified in the context 
        response = "This is the Response for CreditClassifyEvent"
        return CreditClassifyEvent(result=response)
    
    @step
    async def debit_classify(self, ctx: Context, ev: TriggerDebit) -> DebitClassifyEvent:
        
        #Get the Dataframe in the Context
        document = await ctx.store.get("document")
        #Filter the Debit transaction using the Dataframe and Use LLM to classify the Event 
        #Store the Dataframe for debit transaction classified in the context 
        response = "This is the Response for DebitClassifyEvent"
        return DebitClassifyEvent(result=response)

    @step
    async def join_for_surplus(self, ctx: Context, ev: CreditClassifyEvent | DebitClassifyEvent) -> TriggerSurplus | TriggerDTI | TriggerBehavioral | None:
        results = ctx.collect_events(ev, [CreditClassifyEvent, DebitClassifyEvent])
        if results is None:
            return None
        credit_event, debit_event = results
        
        #Storing the Classification results in the Context
        await ctx.store.set("credit_classify_result",ev.result)
        await ctx.store.set("debit_classify_result",ev.result)
        
        # Send the Events to Execute Parallely
        ctx.send_event(TriggerSurplus())
        ctx.send_event(TriggerDTI())
        ctx.send_event(TriggerBehavioral())
    
    @step
    async def surplus_analysis(self, ctx: Context, ev: TriggerSurplus) -> SurplusAnalysisEvent:
        #Get the Classify data from the Context
        credit_classify = await ctx.store.get("credit_classify_result")
        debit_classify = await ctx.store.get("debit_classify_result")
        
        #Using Dataframe operation for calculation and use the LLM to Reason over it 
        
        response = "This is the Response for Surplus Analysis Event"
        return SurplusAnalysisEvent(result=response)
    
    @step
    async def dti_analysis(self, ctx: Context, ev: TriggerDTI) -> DebtToIncomeEvent:
        #Get the Classify data from the Context
        credit_classify = await ctx.store.get("credit_classify_result")
        debit_classify = await ctx.store.get("debit_classify_result")
        
        #Using Dataframe operation for calculation and use the LLM to Reason over it 
        
        response = "This is the Response for Debit To Income Analysis Event"
        return DebtToIncomeEvent(result=response)
    
    @step
    async def behavioral_analysis(self, ctx: Context, ev: TriggerBehavioral) -> BehavioralScoreEvent:
        #Get the Classify data from the Context
        credit_classify = await ctx.store.get("credit_classify_result")
        debit_classify = await ctx.store.get("debit_classify_result")
        
        #Using Dataframe operation for calculation and use the LLM to Reason over it 
        
        response = "This is the Response for Debit To Behavioral Analysis & Score Event"
        return BehavioralScoreEvent(result=response)
        
    @step
    async def report_generation(self, ctx: Context, ev: SurplusAnalysisEvent | DebtToIncomeEvent | BehavioralScoreEvent) -> MyStopEvent | None:
        # Waiting all the analysis events to collect 
        data = ctx.collect_events(ev, [SurplusAnalysisEvent, DebtToIncomeEvent, BehavioralScoreEvent])
        
        # check if we can run
        if data is None:
            return None

        # unpack -- data is returned in order
        surplus_event, debt_to_income_event, behavioral_score_event = data
        
        # Need to Modify the Code as per the prompt and LLM Call 
        prompt = f"Give a thorough analysis and generate report on the following questions mentioned for a particular customer Bank Statement and provide the confidence score on your analysis{surplus_event.result}, {debt_to_income_event.result}, {behavioral_score_event.result}"
        #response = await self.llm.acomplete(prompt)
        response = "Report Generated"
        return MyStopEvent(report=response)
    
    

    



# Running the workflow
#workflow_run = BankStatementAnalyzer(timeout=60, verbose=False)
#result = await workflow_run.run(document="Bank statement content goes here.")
#print(str(result))
