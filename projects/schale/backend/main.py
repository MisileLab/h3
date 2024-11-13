from fastapi import FastAPI

from routers.theresa import router as theresa

f = FastAPI()
f.include_router(theresa, prefix="/theresa")

