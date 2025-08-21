from fastapi import APIRouter, WebSocket
from pathlib import Path
import pandas as pd
import os
import asyncio

from controllers.workflow_definition import BankStatementAnalyzer
from llama_index.utils.workflow import (
    draw_all_possible_flows,
    draw_most_recent_execution,
)

BASE_DIR = Path(__file__).resolve().parent.parent
FINAL_CSV_DIR = f"{BASE_DIR}/uploads/final_csv_uploads"
os.makedirs(FINAL_CSV_DIR, exist_ok=True)

router = APIRouter()

@router.websocket("/ws/run")
async def run_bank_statement_workflow(websocket: WebSocket):
    await websocket.accept()
    try:
        # 1. Locate most recent uploaded file
        files = sorted(Path(FINAL_CSV_DIR).iterdir(), key=os.path.getmtime, reverse=True)
        if not files:
            await websocket.send_json({"event": "error", "data": f"No file found in {FINAL_CSV_DIR}"})
            await websocket.close()
            return

        csv_file = files[0]
        await websocket.send_json({"event": "info", "data": f"Processing file: {csv_file.name}"})

        # 2. Read and clean dataframe
        df = pd.read_csv(csv_file, parse_dates=['TransactionDate'])
        await websocket.send_json({"event": "info", "data": f"Loaded {len(df)} rows"})
        await websocket.send_json({"event": "debug", "data": f"Head: {df.head().to_dict()}"})

        # Data type cleanups, stepwise
        if "TransactionDate" in df.columns:
            df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce', dayfirst=True)
            await websocket.send_json({"event": "step", "data": "TransactionDate column parsed to datetime"})
        if "Credit" in df.columns:
            df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce').astype(float).fillna(0.0)
            await websocket.send_json({"event": "step", "data": "Credit column parsed to float"})
        if "Debit" in df.columns:
            df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce').astype(float).fillna(0.0)
            await websocket.send_json({"event": "step", "data": "Debit column parsed to float"})
        if "Balance" in df.columns:
            df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce').astype(float).fillna(0.0)
            await websocket.send_json({"event": "step", "data": "Balance column parsed to float"})
        await websocket.send_json({"event": "info", "data": f"Data preparation complete. Running workflow..."})
        await asyncio.sleep(0.3)

        # 3. Setup workflow to emit events to websocket
        async def emitter(event_name, payload):
            await websocket.send_json({"event": event_name, "data": payload})

        workflow_run = BankStatementAnalyzer(timeout=300, verbose=False)
        workflow_run._emitter = emitter  # Let workflow steps call: self.emit(event_name, payload)

        # 4. Run workflow, steps should emit their own events (by calling self.emit in each @step)
        result = await workflow_run.run(document=df)
        await websocket.send_json({"event": "info", "data": "Workflow run completed successfully."})
        await websocket.send_json({"event": "report", "data": result.report})

        # 5. Optional: Draw flows after report
        draw_all_possible_flows(BankStatementAnalyzer, filename="bank_statement_flow_all.html")
        draw_most_recent_execution(workflow_run, filename="bank_statement_flow_recent.html")
        await websocket.send_json({"event": "done", "data": "Report and workflow diagrams generated."})

    except Exception as e:
        await websocket.send_json({"event": "error", "data": str(e)})
        await websocket.close()
