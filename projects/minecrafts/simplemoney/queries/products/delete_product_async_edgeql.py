# AUTOGENERATED FROM 'queries/products/delete_product.edgeql' WITH:
#     $ edgedb-py --dir queries --no-skip-pydantic-validation


from __future__ import annotations
import dataclasses
import edgedb
import uuid


@dataclasses.dataclass
class DeleteProductResult:
    id: uuid.UUID


async def delete_product(
    executor: edgedb.AsyncIOExecutor,
    *,
    bank_name: str,
    name: str,
) -> list[DeleteProductResult]:
    return await executor.query(
        """\
        delete ((select Bank {products: {name}} filter .name = <str>$bank_name and .products.name = <str>$name).products)\
        """,
        bank_name=bank_name,
        name=name,
    )