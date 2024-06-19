from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
async def test():
  return JSONResponse(content={"test": "test"})
