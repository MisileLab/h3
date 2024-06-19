from fastapi import FastAPI
from fastapi.responses import RedirectResponse

app = FastAPI()

@app.get("/")
async def test() -> RedirectResponse:
  return RedirectResponse("https://t.ly/-7hpd")
