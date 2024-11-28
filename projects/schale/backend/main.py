from routers.theresa import router as theresa

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

f = FastAPI(root_path="/api")
f.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
f.include_router(theresa, prefix="/theresa")
