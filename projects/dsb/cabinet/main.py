from googlesearch import search # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from os import getenv

proxy_url = getenv("PROXY_URL")
proxy_user = getenv("PROXY_USERNAME")
proxy_pass = getenv("PROXY_PASSWORD")

if None in [proxy_url, proxy_user, proxy_pass]:
  proxy = None
else:
  proxy = f"http://{proxy_user}:{proxy_pass}@{proxy_url}"
print(proxy)

base_query = "site:x.com"
suicidal = base_query + " 자살"

data_num = 100

result = list(search(suicidal, advanced=True, unique=True, num_results=data_num, sleep_interval=5, lang="ko", safe=None, ssl_verify=None, proxy=proxy)) # pyright: ignore[reportArgumentType]
for i in result:
  print(i.title) # pyright: ignore[reportAny]
