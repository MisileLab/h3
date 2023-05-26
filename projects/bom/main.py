# Pressure Redstone
# (Mekanism/Pneumatic) input/output, usage
# Pneumatic bar

from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/")
async def client_websocket(ws: WebSocket):
    await ws.accept()
    while True:
        data = ws.receive_json()
        await ws.send_json(data)
