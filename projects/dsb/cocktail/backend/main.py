from fastapi import FastAPI, HTTPException, status, WebSocket
from orjson import dumps

from secrets import SystemRandom
from pathlib import Path

app = FastAPI()
CHECKPOINT = "it_needs_to_be_changed"
VERSION = "Midori Sour"
websockets = []
chats = []
CHECKPOINT = "test"

class User:
 def __init__(self, gpg: str, name: str, ws: WebSocket):
  self.ws = ws
  self.name = name

if CHECKPOINT == "it_needs_to_be_changed":
 print("plz return")
 exit(1)

@app.get("/")
async def ping():
 return VERSION

@app.websocket("/chat/{PW}")
async def chat(ws: WebSocket, PW: str):
 if PW != CHECKPOINT:
  raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
 await ws.accept()
 try:
  while True:
   data = await ws.receive_json()
   for i in websockets:
    i.ws.send_json({"msg": data["msg"], "name": data["name"]})
    chats.append({"msg": data["msg"], "name": data["name"]})
 except WebsocketDisconnect:
  websockets.remove(ws)

 @app.get("/save/{PW}")
 async def save(PW: str):
  if PW != CHECKPOINT:
   raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
  Path(f"/persist/chats/data_{SystemRandom().randint(1, 100_00_000_00000)}").write_text(chats)

