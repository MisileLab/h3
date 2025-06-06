# AUTOGENERATED FROM 'queries/bank/get_bank_by_id.edgeql' WITH:
#     $ gel-py --dir queries --no-skip-pydantic-validation


from __future__ import annotations
import dataclasses
import gel
import uuid


@dataclasses.dataclass
class GetBankByIdResult:
    id: uuid.UUID
    name: str


async def get_bank_by_id(
    executor: gel.AsyncIOExecutor,
    *,
    id: uuid.UUID,
) -> GetBankByIdResult | None:
    return await executor.query_single(
        """\
        select default::Bank { name } filter .id = <uuid>$id\
        """,
        id=id,
    )
