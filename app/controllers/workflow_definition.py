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

import pandas as pd

from typing import Dict, Any, List, Union
from . import steps

#Langchain Imports 
from langchain.schema.runnable import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI







class MyCustomStartEvent(StartEvent):
    #document: dataframe
    document: pd.DataFrame  # Assuming the document is a DataFrame
    
    
class TriggerCredit(Event): pass
class TriggerDebit(Event): pass

class TriggerSurplus(Event): pass
class TriggerDTI(Event): pass 
class TriggerBehavioral(Event): pass
    
class CreditClassifyEvent(Event):
    #result: 
    result: pd.DataFrame   # Assuming the result is a dictionary with classification results
    
class DebitClassifyEvent(Event):
    #result:
    result: pd.DataFrame  # Assuming the result is a dictionary with classification results
    
class SurplusAnalysisEvent(Event):
    #result:
    result: Dict[str, Any]  # Assuming the result is a dictionary with surplus analysis results
    
class DebtToIncomeEvent(Event):
    #result:
    result: Dict[str, Any]  # Assuming the result is a dictionary with debt-to-income analysis results
    
class BehavioralScoreEvent(Event):
    #result:
    result: Dict[str, Any]  # Assuming the result is a dictionary with behavioral score analysis results    
    
    

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
        response = steps.credit_analysis(document)
        # Assuming response is a dict or similar result from the LLM
        await ctx.store.set("credit_classify_result", response)
        #response = [{"Result":"This is the Response for CreditClassifyEvent"}]
        print("Credit Classification Result:", response)
        return CreditClassifyEvent(result=response)
    
    @step
    async def debit_classify(self, ctx: Context, ev: TriggerDebit) -> DebitClassifyEvent:
        
        #Get the Dataframe in the Context
        document = await ctx.store.get("document")
        #Filter the Debit transaction using the Dataframe and Use LLM to classify the Event 
        #Store the Dataframe for debit transaction classified in the context 
        response = steps.debit_analysis(document)
        await ctx.store.set("debit_classify_result", response)
        #response = "This is the Response for DebitClassifyEvent"
        print("Debit Classification Result:", response)
        return DebitClassifyEvent(result=response)

    @step
    async def join_for_surplus(self, ctx: Context, ev: CreditClassifyEvent | DebitClassifyEvent) -> TriggerSurplus | TriggerDTI | TriggerBehavioral | None:
        results = ctx.collect_events(ev, [CreditClassifyEvent, DebitClassifyEvent])
        if results is None:
            return None
        credit_event, debit_event = results
        
        #Storing the Classification results in the Context
        #await ctx.store.set("credit_classify_result",ev.result)
        #await ctx.store.set("debit_classify_result",ev.result)
        
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
        # For example, calculating surplus from credit and debit classifications
        response = steps.surplus_commentry(credit_classify, debit_classify)
        # Store the result in the context
        print("Surplus Analysis Result:", response)
        await ctx.store.set("surplus_analysis_result", response)
        #response = "This is the Response for Surplus Analysis Event"
        return SurplusAnalysisEvent(result=response)
    
    @step
    async def dti_analysis(self, ctx: Context, ev: TriggerDTI) -> DebtToIncomeEvent:
        #Get the Classify data from the Context
        credit_classify = await ctx.store.get("credit_classify_result")
        debit_classify = await ctx.store.get("debit_classify_result")
        
        #Using Dataframe operation for calculation and use the LLM to Reason over it
        # For example, calculating debt-to-income ratio from credit and debit classifications
        response = steps.dti_commentry(credit_classify, debit_classify)
        await ctx.store.set("dti_analysis_result", response)
        
        #response = "This is the Response for Debit To Income Analysis Event"
        return DebtToIncomeEvent(result=response)
    
    @step
    async def behavioral_analysis(self, ctx: Context, ev: TriggerBehavioral) -> BehavioralScoreEvent:
        #Get the Classify data from the Context
        credit_classify = await ctx.store.get("credit_classify_result")
        debit_classify = await ctx.store.get("debit_classify_result")
        
        #Using Dataframe operation for calculation and use the LLM to Reason over it
        # For example, calculating behavioral score from credit and debit classifications
        response = steps.behavioral_commentry(credit_classify, debit_classify)
        await ctx.store.set("behavioral_analysis_result", response)
        
        #response = "This is the Response for Behavioral Analysis & Score Event"
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
        # Generate the report based on the analysis results
        # Note: This is a placeholder for the actual report generation logic
        print("Surplus Analysis Result:", surplus_event.result["result"])
        print("Debt to Income Analysis Result:", debt_to_income_event.result["result"])
        print("Behavioral Score Analysis Result:", behavioral_score_event.result["result"])        
        # Need to Modify the Code as per the prompt and LLM Call 
        #prompt = f"Give a thorough analysis and generate report on the following questions mentioned for a particular customer Bank Statement and provide the confidence score on your analysis{surplus_event.result}, {debt_to_income_event.result}, {behavioral_score_event.result}"
        with open('app/prompts/report_generation.txt', 'r', encoding='utf-8') as f:
            user_prompt_template = f.read()
        prompt = PromptTemplate.from_template(user_prompt_template)
        llm = ChatOpenAI(
        model="o3-mini",
        #temperature=0.2,
        reasoning_effort="medium",   # constrain reasoning effort (o-series models)
    )
        chain = prompt | llm
        response = await chain.ainvoke({
            "surplus_analysis":surplus_event.result["result"],
            "dti_analysis":debt_to_income_event.result["result"],
            "behavior_analysis":behavioral_score_event.result["result"]}
        )
        response = response.content
        #response = await self.llm.acomplete(prompt)
        print("Generated Report:", response)
        return MyStopEvent(report=response)
    
    

    



# Running the workflow
#workflow_run = BankStatementAnalyzer(timeout=60, verbose=False)
#result = await workflow_run.run(document="Bank statement content goes here.")
#print(str(result))
