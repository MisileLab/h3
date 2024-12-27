from os import getenv

def get_proxy():
  proxy_url = getenv("PROXY_URL")
  proxy_user = getenv("PROXY_USERNAME")
  proxy_pass = getenv("PROXY_PASSWORD")

  if None in [proxy_url, proxy_user, proxy_pass]:
    proxy = None
  else:
    proxy = f"http://{proxy_user}:{proxy_pass}@{proxy_url}"
  return proxy

