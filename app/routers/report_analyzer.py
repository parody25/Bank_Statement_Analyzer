import uuid
from typing import List, Optional

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

# Import necessary modules to run the Event Driven Workflow
from controllers import run
from utils.pdf_report_generation import create_templated_pdf


router = APIRouter(prefix="/analyzer", tags=["Report Analyzer"])

@router.post("/run") # Removed response_model for more flexible dict response
async def run_workflow():
    """
    Run the report analyzer workflow, save the result as a templated PDF,
    and return the structured analysis result as JSON.
    """
    try:
        run_id = str(uuid.uuid4())
        
        # The 'run()' function now returns a structured dictionary
        structured_result = await run()
        
        if not structured_result or not structured_result.get("sections"):
            raise HTTPException(status_code=404, detail="Analysis returned no content.")

        # --- ACTION: Generate the templated PDF ---
        try:
            create_templated_pdf(report_data=structured_result, run_id=run_id)
        except Exception as pdf_error:
            # Log the error but don't fail the request
            print(f"WARNING: PDF generation failed. Error: {pdf_error}")
        
        # Return the new structured result as a JSON response
        return JSONResponse(content={"run_id": run_id, "result": structured_result})
    
    except Exception as e:
        # Handle workflow exceptions
        raise HTTPException(status_code=500, detail=str(e))



