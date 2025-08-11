# import uuid
# from typing import List, Optional

# from fastapi import APIRouter, Request, HTTPException
# from fastapi.responses import JSONResponse

# # Import necessary modules to run the Event Driven Workflow
# from controllers import run


# router = APIRouter(prefix="/analyzer", tags=["Report Analyzer"])

# @router.post("/run", response_model=Optional[str])
# async def run_workflow():
#     """
#     Run the report analyzer workflow.
#     """
#     try:
#         # Generate a unique identifier for the workflow run
#         run_id = str(uuid.uuid4())
        
#         # Run the workflow and get the result
#         result = await run()
        
#         # Return the result as a JSON response
#         return JSONResponse(content={"run_id": run_id, "result": result})
    
#     except Exception as e:
#         # Handle any exceptions that occur during the workflow run
#         raise HTTPException(status_code=500, detail=str(e))


import os
import glob
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from controllers import run

router = APIRouter(prefix="/analyzer", tags=["Report Analyzer"])

DATA_DIR = "app/uploads/final_csv_uploads"

@router.post("/run")
async def run_workflow():
    try:
        run_id = str(uuid.uuid4())

        # Find the latest UUID folder or file set
        credit_files = sorted(glob.glob(os.path.join(DATA_DIR, "credit_*.csv")), key=os.path.getmtime)
        debit_files = sorted(glob.glob(os.path.join(DATA_DIR, "debit_*.csv")), key=os.path.getmtime)
        combined_files = sorted(glob.glob(os.path.join(DATA_DIR, "combined_*.csv")), key=os.path.getmtime)

        if not (credit_files and debit_files and combined_files):
            raise FileNotFoundError("Required CSV files not found.")

        credit_csv_path = credit_files[-1]
        debit_csv_path = debit_files[-1]
        document_text = open(combined_files[-1], "r").read()

        result = await run(
            document_text=document_text,
            credit_csv_path=credit_csv_path,
            debit_csv_path=debit_csv_path
        )

        return JSONResponse(content={"run_id": run_id, "result": result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


