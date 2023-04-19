from fastapi import FastAPI, HTTPException
from os import makedirs
from os.path import isdir, isfile

if not isdir("files"):
    makedirs("files")

app = FastAPI()

@app.get("/{file}")
def file_get(file: str):
    if not isfile(f"files/{file}"):
        raise HTTPException(status_code=404)
    with open(f'files/{file}', 'rb') as r:
        return r.read()

@app.post("/{file}")
def file_post(
    file: str,
    content: bytes
):
    if isfile(f"files/{file}"):
        raise HTTPException(status_code=409)
    with open(f'files/{file}', 'wb') as w:
        w.write(content)
