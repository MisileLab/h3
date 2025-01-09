with
  receiver := (select User filter .userid = <int64>$receiver_id),
  sender := (select Bank filter .name = <str>$bank_name),
  loan := (select Loan filter .sender = <uuid>sender.id and .receiver = <uuid>receiver.id and .product.name = <str>$product_name),
update loan set {amount := <int64>$amount};
update Bank set {money := <int64>$bank_money};
update User filter .userid = <int64>$receiver_id set {money := <int64>$receiver_money};
delete Loan filter .amount = 0;