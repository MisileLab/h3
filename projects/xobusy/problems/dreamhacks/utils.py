from bs4 import Tag, BeautifulSoup

def verify(v: BeautifulSoup | Tag | None, k: str) -> BeautifulSoup | Tag:
  realv = getattr(v, k)
  if realv is None:
    raise TypeError()
  return realv
