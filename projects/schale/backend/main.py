from fastapi import FastAPI

from routers.theresa import router as theresa

f = FastAPI(root_path="/api")
f.include_router(theresa, prefix="/theresa")
