# AUTOGENERATED FROM 'queries/loan/pay_loan.edgeql' WITH:
#     $ edgedb-py --dir queries --no-skip-pydantic-validation


from __future__ import annotations
import dataclasses
import edgedb
import uuid


@dataclasses.dataclass
class PayLoanResult:
    id: uuid.UUID


async def pay_loan(
    executor: edgedb.AsyncIOExecutor,
    *,
    receiver_id: int,
    bank_name: str,
    product_name: str,
    amount: int,
    fee: int,
) -> list[PayLoanResult]:
    return await executor.query(
        """\
        with
          receiver := (select User filter .userid = <int64>$receiver_id),
          sender := (select Bank filter .name = <str>$bank_name),
          loan := (
            select Loan
            filter .sender = sender.id and .receiver = receiver.id and .product.name = <str>$product_name
          ),
          computed := (<int64>$amount - <int64>math::ceil(<int64>$amount / 100 * <int64>$fee)),
          def := (update loan set {amount := .amount - <int64>$amount}),
          def2 := (update sender set {money := .money + <int64>$amount}),
          def3 := (update receiver set {
            money := .money - computed
          }),
          def4 := (delete Loan filter .amount = 0)
        select {def, def2, def3, def4};\
        """,
        receiver_id=receiver_id,
        bank_name=bank_name,
        product_name=product_name,
        amount=amount,
        fee=fee,
    )