from dataclasses import dataclass
from typing import final

@final
@dataclass
class Error:
  detail: str | None

def generate_error_responses(
  status_codes: set[int],
  has_parameters: bool = True
) -> dict[int | str, dict[str, type[Error]]]:
  if has_parameters:
    status_codes.add(422)
  return {k: {"model": Error} for k in status_codes}

