from fastapi import APIRouter, WebSocket
from pathlib import Path
import pandas as pd
import os
import asyncio
import uuid  # NEW: To generate unique IDs for reports

from controllers.workflow_definition import BankStatementAnalyzer
from llama_index.utils.workflow import (
    draw_all_possible_flows,
    draw_most_recent_execution,
)
# NEW: Import our PDF generation utility
from utils.pdf_report_generation import create_templated_pdf

BASE_DIR = Path(__file__).resolve().parent.parent
FINAL_CSV_DIR = f"{BASE_DIR}/uploads/final_csv_uploads"
os.makedirs(FINAL_CSV_DIR, exist_ok=True)

router = APIRouter()

@router.websocket("/ws/run")
async def run_bank_statement_workflow(websocket: WebSocket):
    await websocket.accept()
    
    # NEW: Generate a unique run_id at the start of the process.
    # This will be used for the PDF filename.
    run_id = str(uuid.uuid4())
    await websocket.send_json({"event": "info", "data": f"Workflow started with run ID: {run_id}"})
    
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
        # await websocket.send_json({"event": "debug", "data": f"Head: {df.head().to_dict()}"})

        # Data type cleanups (no changes here)
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

        # 3. Setup workflow to emit events to websocket (no changes here)
        async def emitter(event_name, payload):
            await websocket.send_json({"event": "event", "name": event_name, "data": payload})

        workflow_run = BankStatementAnalyzer(timeout=300, verbose=False)
        workflow_run._emitter = emitter

        # 4. Run workflow
        result = await workflow_run.run(document=df)
        # 4a. The result.report is now a structured dictionary.
        structured_result = result.report
        
        await websocket.send_json({"event": "info", "data": "Workflow run completed successfully."})
        
        # 4b. Send the full structured report to the UI. The UI can now render this properly.
        await websocket.send_json({"event": "report", "data": structured_result})

        # 5. NEW: Generate the PDF report on the server.
        try:
            await websocket.send_json({"event": "info", "data": "Generating PDF report..."})
            
            # Call our existing PDF templating function
            create_templated_pdf(report_data=structured_result, run_id=run_id)
            
            pdf_filename = f"{run_id}.pdf"
            await websocket.send_json({
                "event": "pdf_generated", 
                "data": {
                    "message": f"Successfully generated PDF report: {pdf_filename}",
                    "filename": pdf_filename,
                    "run_id": run_id
                }
            })
        except Exception as pdf_error:
            # Inform the UI that PDF generation failed, but don't kill the connection.
            # The main analysis was still a success.
            await websocket.send_json({
                "event": "error", 
                "data": f"Analysis succeeded, but PDF generation failed: {pdf_error}"
            })
            
        # 6. Optional: Draw flows after report (no changes here)
        draw_all_possible_flows(BankStatementAnalyzer, filename="bank_statement_flow_all.html")
        draw_most_recent_execution(workflow_run, filename="bank_statement_flow_recent.html")
        await websocket.send_json({"event": "done", "data": "Report and workflow diagrams generated."})

    except Exception as e:
        # Send a final error message if the whole process fails
        await websocket.send_json({"event": "error", "data": f"A critical error occurred: {e}"})
    finally:
        # try to close the connection gracefully
        await websocket.close()