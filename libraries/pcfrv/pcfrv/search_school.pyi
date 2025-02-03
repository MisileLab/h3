from dataclasses import dataclass

@dataclass
class School:
    name: str
    period: str
    school_code: int
    period_code: str
    def __init__(self, name, period, school_code, period_code) -> None: ...

comcigan_url: str

def get_code() -> str: ...
def get_school_code(school_name: str) -> list[School]: ...
