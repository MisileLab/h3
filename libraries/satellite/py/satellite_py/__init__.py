from dataclasses import dataclass # noqa: E402
from typing import final  # noqa: E402

@final
@dataclass
class Error:
  detail: str | None

def generate_error_responses(status_codes: list[int]) -> dict[int | str, dict[str, type[Error]]]:
  return {k: {"model": Error} for k in status_codes}

