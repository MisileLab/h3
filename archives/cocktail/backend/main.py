from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from orjson import dumps
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES

from secrets import SystemRandom
from pathlib import Path
import os
from ast import literal_eval as ev

app = FastAPI()
app.add_middleware(
 CORSMiddleware,
 allow_origins=["*"],
 allow_credentials=True,
 allow_methods=["*"],
 allow_headers=["*"]
)
# it needs to be change
CHECKPOINT = "test"
VERSION = "Midori Sour"
websockets = []
chats = []

def encrypt(con: str | bytes, pk: bytes):
 c = AES.new(pk, AES.MODE_EAX)
 nonce = c.nonce
 ciphertext, tag = c.encrypt_and_digest(con)
 return (nonce, ciphertext, tag)

def decrypt(pk: bytes, ciphertext: bytes, nonce: str, tag: str):
 cipher = AES.new(pk, AES.MODE_EAX, nonce=nonce)
 plaintext = cipher.decrypt(ciphertext)
 try:
  cipher.verify(tag)
  return plaintext
 except ValueError:
  print("Key incorrect or message corrupted")

# you can use decrypt function for decrypting chats
# gpgs = [
#   97757888275179,
#   [
#     "b'%?\\xa5\\x00|\\xd5\\x87.\\xfaiFT\\xb7\\xacba'"
#   ],
#   [
#     [
#       "b'\\xfe\\xe5^a<\\xda(\\xe3\\x1f\\x98\\xc4V[t\\x80\\x9f'",
#       "b'\\xb7 \\x87\\xb4\\xf7q\\xf7\\x87\\xd5\\xf9\\x0c\\x94\\xa5(\\xb9\\xb9\\xd4(K\\xb1\\xdas\\x91i\\xe4KQ*\\n\\xd6\\x1b~\\xd9\\xffF\\x84\\x1e+\\xb1\\x94g=\\x10\\x1d;\\xaf\\x08\\xc6\\xa8\\n\\xee.\\xa9\\xbb\\x8e\\x9d>\\t%^\\xbcJ'",
#       "b'`\\x1a\\xac\\xbd\\xb6\\xea\\xe0\\xab\\xa0Y\\xb2\\x88\\xea\\x91\\xab\\xda'"
#     ]
#   ]
# ]

# for i, i2 in enumerate(gpgs[1]):
#  gpgs[2][i][1] = decrypt(ev(i2), ev(gpgs[2][i][1]), ev(gpgs[2][i][0]), ev(gpgs[2][i][2]))
# print([i[1].decode('utf8') for i in gpgs[2]])
# exit()

class User:
 def __init__(self, name: str, ws: WebSocket):
  self.ws = ws
  self.name = name

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
   if data["type"] == "chat":
    for i in websockets:
     await i.ws.send_json({"msg": data["msg"], "name": data["name"]})
     chats.append({"msg": data["msg"], "name": data["name"]})
   elif data["type"] == "login":
    websockets.append(User(data["name"], ws))
 except WebSocketDisconnect:
  ia = None
  for i, i2 in enumerate(websockets):
   if i2.ws == ws:
    ia = i
  if ia is not None:
   del websockets[ia]

@app.get("/save/{PW}")
async def save(PW: str):
 if PW != CHECKPOINT:
  raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
 tmp = str(chats).encode('utf8')
 sig = ""
 rnd = SystemRandom().randint(1,100_000_000_000_000)
 keys = []
 metadatas = []
 for i in websockets:
  keys.append(get_random_bytes(16))
  tmp2 = encrypt(tmp, keys[-1])
  metadatas.append(tmp2)
  tmp = tmp2.__getitem__(1)
  sig += f"{i.name}\n"
 Path(f"/persist/chats/data_{rnd}").write_bytes(tmp)
 Path(f"/persist/chats/sig_{rnd}").write_text(sig)
 return (rnd, [str(key) for key in keys], [(str(i) for i in metadata) for metadata in metadatas])
