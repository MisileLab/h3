with
  receiver := (select User filter .userid = <int64>$receiver_id),
  sender := (select Bank filter .name = <str>$bank_name),
  product := (select Product filter .id = <uuid>$product_id),
  loan := (insert Loan {
    amount := <int64>$amount,
    receiver := receiver.id,
    sender := sender.id,
    product := product,
    date := <datetime>$date
  })
update Bank set {loans += loan, money := <int64>$bank_money};
update User filter .userid = <int64>$receiver_id set {money := <int64>$receiver_money};