from edgedb import create_async_client # pyright: ignore[reportUnknownVariableType]
from typing import Final, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
  from _typeshed import DataclassInstance
else:
  DataclassInstance = TypeVar('DataclassInstance')

class DB:
  def __init__(self, *args, **kwargs): # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    self.db: Final = create_async_client(*args, **kwargs) # pyright: ignore[reportUnknownArgumentType]

  async def query(self, query: str, *args, **kwargs) -> list[DataclassInstance]: # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    return await self.db.query(query, *args, **kwargs) # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType, reportUnknownMemberType]

  async def query_single(self, query: str, *args, **kwargs) -> DataclassInstance | None: # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    return await self.db.query_single(query, *args, **kwargs) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportAny]

  async def query_required_single(self, query: str, *args, **kwargs) -> DataclassInstance | None: # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
    return await self.db.query_required_single(query, *args, **kwargs) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportAny]

