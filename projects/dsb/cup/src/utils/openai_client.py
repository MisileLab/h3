"""Shared helpers for calling OpenAI models via the official SDK."""

from __future__ import annotations

from typing import Any

from openai import OpenAI
from pydantic import BaseModel


def call_openai(
  model: str,
  api_key: str,
  messages: list[dict[str, str]],
  response_format: dict[str, Any] | None = None,
) -> str:
  client = OpenAI(api_key=api_key)
  if model.lower().startswith("gpt-5"):
    return _call_responses_api(client, model, messages)
  return _call_chat_api(client, model, messages, response_format)


def _call_chat_api(
  client: OpenAI,
  model: str,
  messages: list[dict[str, str]],
  response_format: dict[str, Any] | None,
) -> str:
  params: dict[str, Any] = {
    "model": model,
    "messages": messages,
  }
  if response_format:
    params["response_format"] = response_format

  response = client.chat.completions.create(**params)
  message = response.choices[0].message
  content = message.content

  if isinstance(content, list):
    # New SDK may return list of ContentPart objects
    texts = [part.text for part in content if getattr(part, "type", None) == "text"]
    if texts:
      return "\n".join(texts)
  return content or ""


def _call_responses_api(
  client: OpenAI,
  model: str,
  messages: list[dict[str, str]],
) -> str:
  input_payload = [
    {
      "role": msg["role"],
      "content": [{"type": "input_text", "text": msg["content"]}],
    }
    for msg in messages
  ]
  response = client.responses.create(
    model=model,
    input=input_payload,
  )
  for item in response.output:
    if not getattr(item, "content", None):
      continue
    for content in item.content:
      if content.type in {"output_text", "text"} and content.text:
        return content.text
  raise RuntimeError("No text output returned from OpenAI Responses API")


def call_openai_structured(
  model: str,
  api_key: str,
  messages: list[dict[str, str]],
  schema: type[BaseModel],
) -> BaseModel:
  client = OpenAI(api_key=api_key)
  input_payload = [
    {"role": msg["role"], "content": msg["content"]}
    for msg in messages
  ]
  response = client.responses.parse(
    model=model,
    input=input_payload,
    text_format=schema,
  )
  return response.output_parsed
