# Pressure Redstone
# (Mekanism/Pneumatic) input/output, usage
# Pneumatic bar
from typing import Annotated
from fastapi import FastAPI, WebSocket, HTTPException, Header

class WebSocketContainer:
    def __init__(self, ws: WebSocket, wstype: str):
        self.ws = ws
        self.wstype = wstype

datas = {}
app = FastAPI()
websockets: dict[WebSocketContainer] = {}

@app.websocket("/")
async def client_websocket(ws: WebSocket):
    await ws.accept()
    data = ws.receive_json()
    websockets[data["type"]] = WebSocketContainer(ws, data["type"])
    datas[data["type"]] = data["data"]
    while True:
        data = ws.receive_json()
        datas[data["type"]] = data["data"]

@app.get("/status/{status}")
async def status_datas(status: str):
    try:
        return datas[status]
    except KeyError:
        return HTTPException(status_code=404)

@app.post("/redstone/{input}")
async def redstone_input(
    redstone_type: Annotated[str | None, Header()] = None, 
    redstone_value: Annotated[bool | None, Header()] = None
):
    if redstone_type is None or redstone_value is None:
        return HTTPException(status_code=400)
    websockets["induction_redstone"].ws.send_json({
        "type": "induction_redstone",
        "redstone_type": redstone_type,
        "redstone_value": redstone_value
    })
