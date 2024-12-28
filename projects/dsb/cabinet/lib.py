from twscrape import API # pyright: ignore[reportMissingTypeStubs]

from os import getenv

def get_proxy():
  proxy_url = getenv("PROXY_URL")
  proxy_user = getenv("PROXY_USERNAME")
  proxy_pass = getenv("PROXY_PASSWORD")

  return (
    None if None in [proxy_url, proxy_user, proxy_pass] else
    f"http://{proxy_user}:{proxy_pass}@{proxy_url}"
  )

async def get_usernames() -> list[str]:
  api = API()
  lst = await api.pool.accounts_info()
  return [a["username"] for a in lst if not (a["active"] or a["logged_in"])]
