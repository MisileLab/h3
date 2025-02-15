# AUTOGENERATED FROM 'queries/credit/get_credit.edgeql' WITH:
#     $ edgedb-py --dir queries --no-skip-pydantic-validation


from __future__ import annotations
import edgedb


async def get_credit(
    executor: edgedb.AsyncIOExecutor,
    *,
    userid: int,
) -> int | None:
    return await executor.query_single(
        """\
        insert User {
          userid := <int64>$userid
        } unless conflict on .userid;
        select (select User {credit} filter .userid = <int64>$userid).credit;\
        """,
        userid=userid,
    )
