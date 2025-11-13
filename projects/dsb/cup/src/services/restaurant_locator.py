"""Utilities for turning restaurant names into Kakao search records."""

from __future__ import annotations

import asyncio
from typing import Iterable

from rich.console import Console
from pydantic import BaseModel

from ..core.kakao_api import search_places
from ..core.types import RestaurantRecord
from ..utils.openai_client import call_openai_structured


console = Console()


class MerchantList(BaseModel):
  merchants: list[str]


class MerchantFilter:
  """Filter candidate strings down to actual restaurants using GPT-5 structured output."""

  MODEL = "gpt-5-mini"

  def __init__(self, api_key: str | None) -> None:
    self.api_key = api_key

  def filter_merchants(self, candidates: list[str]) -> list[str]:
    trimmed = [name.strip() for name in candidates if name.strip()]
    if not trimmed:
      return []

    if not self.api_key:
      console.print(
        "⚠️ OPENAI_API_KEY missing; using unfiltered merchant list.", style="yellow"
      )
      return trimmed

    try:
      messages = self._build_messages(trimmed)
      parsed = call_openai_structured(
        model=self.MODEL,
        api_key=self.api_key,
        messages=messages,
        schema=MerchantList,
      )
      filtered = [name.strip() for name in parsed.merchants if name.strip()]
      console.print(
        f"🤖 Merchant filter kept {len(filtered)} of {len(trimmed)} candidates.",
        style="blue",
      )
      return filtered
    except Exception as e:  # pragma: no cover - network/JSON issues
      console.print(f"⚠️ Merchant filtering failed: {e}. Using raw names.", style="yellow")
      return trimmed

  def _build_messages(self, names: list[str]) -> list[dict[str, str]]:
    numbered = "\n".join(f"{idx+1}. {name}" for idx, name in enumerate(names))
    system_prompt = (
      "You are a data cleaner. From government expense records, extract only"
      " real restaurants, cafes, or food-serving businesses."
      " Ignore departments, job titles, payment methods, events, or numeric"
      " descriptions. Return JSON matching the schema {merchants: list[str]}"
      " where each merchant is a single string. Keep original spelling/order"
      " and drop duplicates."
    )
    user_prompt = (
      "Filter the following list and return only real restaurant/store names:"
      f"\n\n{numbered}\n\nReturn JSON only."
    )
    return [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_prompt},
    ]


class RestaurantLocator:
  """Look up restaurant metadata via Kakao Local API."""

  def __init__(
    self,
    api_key: str | None,
    openai_api_key: str | None = None,
    ref_wtm_x: float | None = None,
    ref_wtm_y: float | None = None,
  ) -> None:
    self.api_key = api_key
    self.ref_wtm_x = ref_wtm_x
    self.ref_wtm_y = ref_wtm_y
    self.merchant_filter = MerchantFilter(openai_api_key)

  def lookup(
    self,
    names: Iterable[str],
    source_pdf: str,
    source_url: str | None = None,
  ) -> list[RestaurantRecord]:
    """Synchronously perform lookup for the provided names."""
    unique = self._dedup_names(names)
    if not unique:
      return []

    filtered = self.merchant_filter.filter_merchants(unique)
    if not filtered:
      console.print("⚠️ Merchant filter returned no candidates.", style="yellow")
      return []

    return asyncio.run(
      self._lookup_async(filtered, source_pdf=source_pdf, source_url=source_url)
    )

  async def _lookup_async(
    self,
    names: list[str],
    source_pdf: str,
    source_url: str | None,
  ) -> list[RestaurantRecord]:
    records: list[RestaurantRecord] = []

    for name in names:
      address = None
      x = None
      y = None

      if self.api_key:
        try:
          result = await search_places(
            self.api_key,
            name,
            nearest_only=True,
            ref_wtm_x=self.ref_wtm_x,
            ref_wtm_y=self.ref_wtm_y,
          )
          if result.documents:
            place = result.documents[0]
            address = place.road_address_name or place.address_name or place.place_name
            x = float(place.x)
            y = float(place.y)
            console.print(f"✅ Kakao match for '{name}': {address}", style="green")
          else:
            console.print(f"⚠️ No Kakao results for '{name}'", style="yellow")
        except Exception as e:  # pragma: no cover - network issues
          console.print(f"⚠️ Kakao lookup failed for '{name}': {e}", style="yellow")
      else:
        console.print(
          f"⚠️ KAKAO_API_KEY missing; storing '{name}' without coordinates.",
          style="yellow",
        )

      records.append(
        RestaurantRecord(
          name=name,
          address=address,
          x=x,
          y=y,
          source_pdf=source_pdf,
          source_url=source_url,
        )
      )

    return records

  def _dedup_names(self, names: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
      candidate = name.strip()
      if not candidate:
        continue
      if candidate in seen:
        continue
      seen.add(candidate)
      ordered.append(candidate)
    return ordered
