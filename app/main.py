from pathlib import Path
from fastapi.templating import Jinja2Templates
import uvicorn
from fastapi import FastAPI, Request
from routers import api_router
from fastapi.responses import HTMLResponse

BASE_DIR = Path(__file__).resolve().parent  # Adjust as needed
TEMPLATES_DIR = BASE_DIR / "templates" 

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app = FastAPI(title="Bank Statement Analyzer Event Driven Workflow")
app.include_router(api_router)


@app.get("/", response_class=HTMLResponse)
async def frontend(request: Request):
    return templates.TemplateResponse("frontend.html", {"request": request})

if __name__ == '__main__':
    uvicorn.run('main:app', host='localhost', port=8000, reload=True)