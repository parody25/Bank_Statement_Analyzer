import uvicorn
from fastapi import FastAPI
from routers import api_router
from fastapi.responses import RedirectResponse


app = FastAPI(title="Bank Statement Analyzer Event Driven Workflow")
app.include_router(api_router)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url='/docs')

if __name__ == '__main__':
    uvicorn.run('main:app', host='localhost', port=8000, reload=True)