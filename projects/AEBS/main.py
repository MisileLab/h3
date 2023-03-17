from fastapi import FastAPI, Header, HTTPException
from json import loads

app = FastAPI()
with open("apikey", "r") as r:
    _apikey = r.read()

@app.get("/reactor")
async def reactor(
    apikey: str = Header(),
    reactor_data: str = Header(),
    turbine: str = Header(),
    coolant_tank: str = Header(),
    heat_tank: str = Header(),
    nuclear_fuel_tank: str = Header(),
    nuclear_tank: str = Header(),
    energy_cube: str = Header()
):
    if _apikey != apikey:
        raise HTTPException(status_code=401, detail="incorrect api key")
    try:
        reactor_data = loads(reactor_data)
        turbine = loads(turbine)
        coolant_tank = loads(coolant_tank)
        heat_tank = loads(heat_tank)
        nuclear_fuel_tank = loads(nuclear_fuel_tank)
        nuclear_tank = loads(nuclear_tank)
        energy_cube = loads(energy_cube)
    except ValueError as e:
        raise HTTPException(status_code=422, detail="can't loads to json") from e
