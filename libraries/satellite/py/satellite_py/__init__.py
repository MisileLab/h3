edge_db = True

from dataclasses import dataclass # noqa: E402
from typing import TYPE_CHECKING, TypeVar, final  # noqa: E402

try:
  from edgedb import create_async_client  # pyright: ignore[reportUnknownVariableType]
except ImportError:
  edge_db = False

if TYPE_CHECKING:
  from _typeshed import DataclassInstance
else:
  DataclassInstance = TypeVar('DataclassInstance')

if edge_db:
  @final
  class DB:
    def __init__(self, *args, **kwargs): # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
      self.db = create_async_client(*args, **kwargs) # pyright: ignore[reportUnknownArgumentType, reportPossiblyUnboundVariable]

    async def query(self, query: str, *args, **kwargs) -> list[DataclassInstance]: # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
      return await self.db.query(query, *args, **kwargs) # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType, reportUnknownMemberType]

    async def query_single(self, query: str, *args, **kwargs) -> DataclassInstance | None: # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
      return await self.db.query_single(query, *args, **kwargs) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportAny]

    async def query_required_single(self, query: str, *args, **kwargs) -> DataclassInstance | None: # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
      return await self.db.query_required_single(query, *args, **kwargs) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportAny]

@final
@dataclass
class Error:
  detail: str | None

def generate_error_responses(status_codes: list[int]) -> dict[int | str, dict[str, type[Error]]]:
  return {k: {"model": Error} for k in status_codes}

