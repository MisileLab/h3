from fastapi import FastAPI, Header

from libraries.pow import generate_challenge, verify_challenge

from typing import Annotated

app = FastAPI()

@app.get("/hash")
async def challenge():
  return generate_challenge(

  )

