# AUTOGENERATED FROM 'queries/bank/modify_bank.edgeql' WITH:
#     $ edgedb-py --dir queries --no-skip-pydantic-validation


from __future__ import annotations
import dataclasses
import edgedb
import uuid


@dataclasses.dataclass
class ModifyBankResult:
    id: uuid.UUID


async def modify_bank(
    executor: edgedb.AsyncIOExecutor,
    *,
    owner: int,
    name: str,
    money: int,
) -> ModifyBankResult:
    return await executor.query_single(
        """\
        with
          owner := (select User filter .userid = <int64>$owner)
        insert Bank {
          name := <str>$name,
          money := <int64>$money,
          owner := owner
        } unless conflict on .name else (
          update Bank set {
          money := <int64>$money,
          owner := owner
        })\
        """,
        owner=owner,
        name=name,
        money=money,
    )