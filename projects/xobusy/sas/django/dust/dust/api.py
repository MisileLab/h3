from pathlib import Path
import requests

key = Path("./key").read_text()

def check_air(name: str) -> dict:
  url = "https://api.weatherapi.com/v1/current.json"
  return requests.get(f"{url}?q={name}&key={key}").json()

