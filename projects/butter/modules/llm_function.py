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
from functools import wraps
from inspect import iscoroutinefunction

middle_prompt = Path("./prompts/middle_prompt").read_text()
llm_mini = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

async def summarize_and_answer(content: str, question: str) -> str:
  finalvalue: list[str] = []
  end = False
  while True:
    _content = content
    if len(_content) >= 16384:
      _content = content[:16384]
      content = content[16384:]
    else:
      end = True
    summarized = "".join(
      f"String {i * 16384}~{(i + 1) * 16384} summarized:\n{i2}"
      for i, i2 in enumerate(finalvalue))
    tmp = (await llm_mini.ainvoke([
      {"role": "system", "content": middle_prompt},
      {"role": "user", "content": f"content: {content}\nquestion: {question}\n{summarized}"}
    ])).content
    if not isinstance(tmp, str):
      logger.error("summarize and answer doesn't return string")
      return "failed"
    finalvalue.append(tmp)
    if end:
      break
  return '\n'.join(f"Part {i+1}:\n{i2}" for i, i2 in enumerate(finalvalue))

def print_it(func):
  @wraps(func)
  async def wrapper(*args, **kwargs):
    logger.debug(f'{func.__name__}, {args}, {kwargs}')
    res = await func(*args, **kwargs)
    logger.debug(f"return {res}")
    logger.debug(f"type is {type(res)}")
    return res

  @wraps(func)
  def non_async_wrapper(*args, **kwargs):
    logger.debug(f'{func.__name__}, {args}, {kwargs}')
    res = func(*args, **kwargs)
    logger.debug(f"return {res}")
    logger.debug(f"type is {type(res)}")
    return res
  return wrapper if iscoroutinefunction(func) else non_async_wrapper

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
  return d if d.tzinfo is not None else d.replace(tzinfo=timezone.utc)

@print_it
async def send_request(url: str, question: str) -> str:
  async with AsyncClient() as c:
    resp = await c.get(url)
  con = text_from_html(resp.content)
  return await summarize_and_answer(con, question)

@print_it
async def send_request_using_browser(url: str, question: str) -> str:
  op = Options()
  op.add_argument("--headless")
  web = Firefox(options=op)
  web.get(url)
  con = text_from_html(web.page_source)
  web.close()
  return await summarize_and_answer(con, question)

@print_it
async def search_internet(query: str, question: str) -> str:
  query = query.replace("\n", "")
  tmp = await AsyncDDGS().atext(query, region="ko-kr", max_results=10)
  res = []
  for i in tmp:
    res.append({"type": "text", "text": f"Title: {i["title"]}\ndescription: {i["body"]}\nhref: {i["href"]}"})
  logger.debug(res)
  f = (await llm_mini.ainvoke([
    {"role": "system", "content": middle_prompt},
    {"role": "user", "content": f"content: {res}\nquestion: {question}"}
  ])).content
  if not isinstance(f, str):
    logger.error("search_internet does not return string")
    return "failed"
  return f

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
