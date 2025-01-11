with
  receiver := (select User filter .userid = <int64>$receiver_id),
  sender := (select Bank filter .name = <str>$bank_name),
  loan := (
    select Loan
    filter .sender = sender.id and .receiver = receiver.id and .product.name = <str>$product_name
  ),
update loan set {amount := .amount - <int64>$amount};
update Bank filter .name = <str>$bank_name set {money := .money + <int64>$amount};
update User filter .userid = <int64>$receiver_id set {
  money := .money - <int64>$amount
};
delete Loan filter .amount = 0;