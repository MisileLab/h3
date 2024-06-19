from typing import Annotated

from fastapi import Cookie, FastAPI

app = FastAPI()

@app.get("/items/")
async def read_items(data: Annotated[str | None, Cookie()] = None):
    return {"data": data}
