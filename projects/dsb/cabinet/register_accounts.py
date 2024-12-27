from csv import DictReader
from asyncio import run

from twscrape import API, set_log_level # pyright: ignore[reportMissingTypeStubs]

from lib import get_proxy

set_log_level("DEBUG")

api = API()
api.proxy = get_proxy()

async def main():
  with open("./accounts.csv", newline="") as f:
    dr = DictReader(f)
    for i in dr:
      await api.pool.add_account(
        i["name"],
        i["password"],
        i["email"],
        i["email_password"],
        None,
        i["ct0"]
      )

if __name__ == "__main__":
  run(main())

