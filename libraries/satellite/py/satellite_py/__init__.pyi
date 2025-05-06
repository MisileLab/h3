from dataclasses import dataclass

@dataclass
class Error:
    detail: str | None
    def __init__(self, detail: str | None) -> None: ...

def generate_error_responses(status_codes: set[int]) -> dict[int | str, dict[str, type[Error]]]: ...
