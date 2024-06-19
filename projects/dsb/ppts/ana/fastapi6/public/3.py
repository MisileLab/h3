from typing import Annotated

from fastapi import FastAPI, Header

app = FastAPI()

@app.get("/items/")
async def read_items(agents: Annotated[list[str], Header()] = []):
    return {"agents": agents}

