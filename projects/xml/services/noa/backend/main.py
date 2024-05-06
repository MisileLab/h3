from httpx import post
from edgedb import create_async_client
from fastapi import FastAPI, Header, HTTPException, Request, status, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from os import listdir, getcwd
from os.path import getsize, join, realpath, abspath, isdir, mkdir
from typing import Annotated
from pathlib import Path
from shutil import copyfileobj

if not isdir("files"):
  mkdir("files")

SCHALE_URL = "https://schale.misile.xyz"
ROOT_PATH = join(getcwd(), "files")
ORIGINS = ["null","*"]

app = FastAPI()
app.add_middleware(
  CORSMiddleware,
  allow_origins=ORIGINS,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"]
)
app.mount("/file", StaticFiles(directory="files"), "files")

db = create_async_client()

def is_safe_path(path, follow_symlinks=True):
  if follow_symlinks:
    return realpath(path).startswith(ROOT_PATH)
  else:
    return abspath(path).startswith(ROOT_PATH)

def verify_jwt(jwtv: str):
  r = post(f"{SCHALE_URL}/verify", headers={'jwt':jwtv})
  if not r.is_success:
    raise HTTPException(status_code=r.status_code)
  return r.text[1:-1]

@app.get("/")
async def get_gpg(request: Request, jwtv: Annotated[str, Header(alias="jwt")] = ""):
  return await db.query_json("select User {groups} filter .userid = <str>$userid", userid=verify_jwt(jwtv))

@app.get("/files")
async def get_files(request: Request, path: Annotated[str, Header()] = "."):
  rpath = join(ROOT_PATH, path)
  if not is_safe_path(rpath):
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
  if not isdir(rpath):
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
  return [{"name": x, "size": getsize(join(ROOT_PATH, path, x)), "dir": isdir(join(ROOT_PATH, path, x))} for x in listdir(rpath)]

@app.post("/uploadfile")
async def upload_file(request: Request, file: UploadFile, jwtv: Annotated[str, Header(alias="jwt")] = "", path: Annotated[str | None, Header()] = None):
  print(path, is_safe_path(join(ROOT_PATH, str(path))))
  if path is None or not is_safe_path(join(ROOT_PATH, path)):
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
  admin_check = (await db.query("select User {admin} filter .userid = <str>$userid", userid=verify_jwt(jwtv)))
  if len(admin_check) == 0 or not admin_check[0].admin:
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
  filepath = Path(join(ROOT_PATH, path, file.filename))
  if filepath.exists():
    raise HTTPException(status_code=status.HTTP_409_CONFLICT)
  filepath.parent.mkdir(parents=True, exist_ok=True)
  with filepath.open("wb") as f:
    copyfileobj(file.file, f)

