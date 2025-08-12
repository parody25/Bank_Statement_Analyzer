import shutil
import uuid
import os
import pandas as pd

from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path

#Temporary import for pre-process 
from utils.document_preprocessing import preprocess_documents, post_process_tables
from utils.transaction_filter import filter_transactions

router = APIRouter(prefix="/upload",tags=["File Upload"])

#UPLOAD_DIR = Path("uploads")
#UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
BASE_DIR: Path = Path(__file__).resolve().parent.parent
UPLOAD_DIR = f"{BASE_DIR}/uploads/document_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
FINAL_CSV_DIR = f"{BASE_DIR}/uploads/final_csv_uploads"
os.makedirs(FINAL_CSV_DIR, exist_ok=True)



@router.post("/file")
async def upload_file(files: List[UploadFile] = File(...)):
    # Validate file type 
    # Parse multiple files sent via multipart/form-data
    allowed_types = ["image/jpeg", "image/png", "application/pdf","application/vnd.ms-excel",  # .xls
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" # .xlsx
    ]
    
    results = []

    for file in files:
        # Validate file type
        if file.content_type not in allowed_types:
            results.append({
                "filename": file.filename,
                "status": "failed",
                "reason": "Invalid file type"
            })
            continue

        # Save file with unique name
        #file_extension = Path(file.filename).suffix
        #saved_filename = f"{uuid.uuid4()}{file_extension}"
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        try:
            with open(file_path,"wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            results.append({
                "filename": file.filename,
                "saved_as": unique_filename,
                "content_type": file.content_type,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "failed",
                "reason": f"Error saving file: {str(e)}"
            })
        
    # Temprorary Step Preprocess documents
    preprocessed_docs = await preprocess_documents(UPLOAD_DIR)
    if not preprocessed_docs: 
        raise HTTPException(status_code=500, detail="No documents were preprocessed successfully.")
    
    # Post-process tables and save to CSV
    combined_df = post_process_tables()
    if not combined_df.empty:
        # Save the combined CSV to the final directory 
        final_csv_path = os.path.join(FINAL_CSV_DIR, f"combined_{uuid.uuid4().hex}.csv") 
        combined_df.to_csv(final_csv_path, index=False)
        print("Final CSV saved at:", final_csv_path) 
    else:
        raise HTTPException(status_code=500, detail="Post-processing of tables failed.") 
    
    try:
        filter_results = filter_transactions(final_csv_path)
    except Exception as e:
        print(f"Error during transaction filtering: {e}")
        filter_results = {"error": str(e)}

    print("Filter results:", filter_results)

    return JSONResponse(
        status_code=207,
        content={
            "results": results,
            "filter_results": filter_results
        }
    )

    #return JSONResponse(content={"results": results}, status_code=200 if all(r["status"] == "success" for r in results) else 400)
    return JSONResponse(status_code=207, content={"results": results})