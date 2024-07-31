from .config import api_key

from langchain_openai import ChatOpenAI

from loguru import logger
from duckduckgo_search import AsyncDDGS
from httpx import AsyncClient
from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup
from bs4.element import Comment
from pydantic import BaseModel, Field

from datetime import datetime, timezone
from pathlib import Path

middle_prompt = Path("./prompts/middle_prompt").read_text()
llm_mini = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

def print_it(func):
  async def wrapper(*args, **kwargs):
    logger.debug(f'{func.__name__}, {args}, {kwargs}')
    res = await func(*args, **kwargs)
    logger.debug(f"return {res}")
    logger.debug(f"type is {type(res)}")
    return res
  return wrapper

def tag_visible(element):
  if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
    return False
  return not isinstance(element, Comment)

def text_from_html(body):
  soup = BeautifulSoup(body, 'lxml')
  texts = soup.findAll(text=True)
  visible_texts = filter(tag_visible, texts)  
  return u" ".join(t.strip() for t in visible_texts)

def get_date_from_string(date: str) -> datetime:
  d = datetime.fromisoformat(date)
  if d.tzinfo is None:
    d = d.replace(tzinfo=timezone.utc)
  return d

@print_it
async def send_request(url: str, question: str) -> str:
  async with AsyncClient() as c:
    resp = await c.get(url)
  con = text_from_html(resp.content)
  if len(con) >= 16385:
    con = con[:16384]
  return (await llm_mini.ainvoke([{"role": "system", "content": middle_prompt},{"role": "user", "content": f"content: {con}\nquestion: {question}"}])).content

@print_it
async def send_request_using_browser(url: str, question: str) -> str:
  op = Options()
  op.add_argument("--headless")
  web = Firefox(options=op)
  web.get(url)
  con = text_from_html(web.page_source)
  web.close()
  if len(con) >= 16385:
    con = con[:16384]
  return (await llm_mini.ainvoke([{"role": "system", "content": middle_prompt},{"role": "user", "content": f"content: {con}\nquestion: {question}"}])).content

@print_it
async def search_internet(query: str, question: str) -> str:
  query = query.replace("\n", "")
  tmp = await AsyncDDGS().atext(query, region="ko-kr", max_results=10)
  res = []
  for i in tmp:
    res.append({"type": "text", "text": f"Title: {i["title"]}\ndescription: {i["body"]}\nhref: {i["href"]}"})
  logger.debug(res)
  return (await llm_mini.ainvoke([{"role": "system", "content": middle_prompt},{"role": "user", "content": f"content: {res}\nquestion: {question}"}])).content

class sendRequestBase(BaseModel):
  """send request to url and return the summarized content with llm, you can send the question to llm."""
  url: str = Field(..., description="url to send request")
  question: str = Field(..., description="question to ask llm")

class sendRequestUsingBrowserBase(BaseModel):
  """same with send_request, but use browser to get the content. should be used when the content is not loaded by http request."""
  url: str = Field(..., description="url to send request")
  question: str = Field(..., description="question to ask llm")

class searchInternetBase(BaseModel):
  """search the internet with query and return the summarized content with llm, you can send the question to llm."""
  query: str = Field(..., description="query to search")
  question: str = Field(..., description="question to ask llm")

functions = {
  "sendRequestBase": sendRequestBase,
  "sendRequestUsingBrowserBase": sendRequestUsingBrowserBase,
  "searchInternetBase": searchInternetBase
}

middle_converting_functions = {
  sendRequestBase: send_request,
  sendRequestUsingBrowserBase: send_request_using_browser,
  searchInternetBase: search_internet
}
