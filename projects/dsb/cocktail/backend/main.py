from fastapi import FastAPI, HTTPException, status, WebSocket

app = FastAPI()
CHECKPOINT = "it_needs_to_be_changed"
VERSION = "Midori Sour"
websockets = []
CHECKPOINT = "test"

class User:
 def __init__(self, gpg: str, name: str, ws: WebSocket):
  self.ws = ws

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
 while True:
  data = await ws.receive_json()
  if data["type"] == "gpg":
   websockets.append(User(data["gpg"], data["name"], ws))
  elif data["type"] == "send":
   for i in websockets:
    i.ws.send_json({"msg": data["msg"], "gpg": data["gpg"], "name": data["name"]})

