# AUTOGENERATED FROM 'queries/bank/get_bank_by_id.edgeql' WITH:
#     $ edgedb-py --dir queries --no-skip-pydantic-validation


from __future__ import annotations
import dataclasses
import edgedb
import uuid


@dataclasses.dataclass
class GetBankByIdResult:
    id: uuid.UUID
    name: str


async def get_bank_by_id(
    executor: edgedb.AsyncIOExecutor,
    *,
    id: uuid.UUID,
) -> GetBankByIdResult | None:
    return await executor.query_single(
        """\
        select default::Bank { name } filter .id = <uuid>$id\
        """,
        id=id,
    )
