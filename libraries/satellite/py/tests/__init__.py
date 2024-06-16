from fastapi import FastAPI

from ..src.satellitepy import RealWildCardCors

app = FastAPI()

app.add_middleware(RealWildCardCors)

@app.get("/")
async def hi():
  return "hi"
