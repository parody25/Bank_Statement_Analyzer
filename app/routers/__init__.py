from fastapi import APIRouter
from . import document_upload, report_analyzer, analysis

api_router = APIRouter(prefix="/api/v1")


api_router.include_router(report_analyzer.router)
api_router.include_router(document_upload.router)
api_router.include_router(analysis.router)


