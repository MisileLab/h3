from pathlib import Path
from typing import Annotated

from pymongo import MongoClient
from bson.objectid import ObjectId
from fastapi import FastAPI, Header, HTTPException

a = MongoClient(Path("MONGO_URI").read_text().strip())["iusearchbtw"]
b = FastAPI()

@b.get("/post/{id}")
def get_post(id: str):
    c = a["posts"].find_one({"_id": ObjectId(id)})
    c["_id"] = str(c["_id"])
    if c is None:
        raise HTTPException(status_code=404, detail="Post not found")
    return c

@b.get("/posts")
def get_posts():
    c = list(a["posts"].find({}))
    for i in c:
        i["_id"] = str(i["_id"])
    return c

@b.get("/comment/{id}")
def get_comment(id: str):
    c = a["comments"].find_one({"_id": ObjectId(id)})
    c["_id"] = str(c["_id"])
    if c is None:
        raise HTTPException(status_code=404, detail="Comment not found")
    return c

@b.post("/post/upload")
def upload_post(
    title: Annotated[str | None, Header()] = None,
    description: Annotated[str | None, Header()] = None,
    author: Annotated[str | None, Header()] = None
):
    if title is None or description is None or author is None:
        raise HTTPException(400, detail="Missing required header")
    a["posts"].insert_one({
        "title": title,
        "description": description,
        "author": author,
        "like": 0,
        "comments": []
    })

@b.post("/comment/upload")
def upload_comment(
    content: Annotated[str | None, Header()] = None,
    post_id: Annotated[str | None, Header()] = None
):
    c = a["posts"].find_one({"_id": ObjectId(post_id)})
    if content is None or post_id is None:
        raise HTTPException(400, detail="Missing required header")
    if c is None:
        raise HTTPException(404, detail="Post not found")
    d = a["comments"].insert_one({"content": content})
    c["comments"].append(str(d.inserted_id))
    a["posts"].update_one({"_id": ObjectId(post_id)}, {
        "$push": {"comments": c["comments"]}
    })

@b.post("/post/{id}/like")
def like_post(id: str):
    c = a["posts"].find_one({"_id": ObjectId(id)})
    if c is None:
        raise HTTPException(status_code=404, detail="Post not found")
    a["posts"].update_one({"_id": ObjectId(id)}, {"$inc": {"like": 1}})
