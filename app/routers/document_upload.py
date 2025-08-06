import shutil
import uuid
import os

from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path

router = APIRouter(prefix="/upload",tags=["File Upload"])

#UPLOAD_DIR = Path("uploads")
#UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
BASE_DIR: Path = Path(__file__).resolve().parent.parent
UPLOAD_DIR = f"{BASE_DIR}/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)



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

    #return JSONResponse(content={"results": results}, status_code=200 if all(r["status"] == "success" for r in results) else 400)
    return JSONResponse(status_code=207, content={"results": results})