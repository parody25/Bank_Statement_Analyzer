import uuid
from typing import List, Optional

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

# Import necessary modules to run the Event Driven Workflow
from controllers import run


router = APIRouter(prefix="/analyzer", tags=["Report Analyzer"])

@router.post("/run", response_model=Optional[str])
async def run_workflow():
    """
    Run the report analyzer workflow.
    """
    try:
        # Generate a unique identifier for the workflow run
        run_id = str(uuid.uuid4())
        
        # Run the workflow and get the result
        result = await run()
        
        # Return the result as a JSON response
        return JSONResponse(content={"run_id": run_id, "result": result})
    
    except Exception as e:
        # Handle any exceptions that occur during the workflow run
        raise HTTPException(status_code=500, detail=str(e))



