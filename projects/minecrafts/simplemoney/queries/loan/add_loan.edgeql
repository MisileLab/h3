with
  receiver := (select User filter .userid = <int64>$receiver_id),
  sender := (select Bank filter .name = <str>$bank_name),
  product := (select Product filter .id = <uuid>$product_id),
  exist_loan := (select Loan filter .receiver = receiver.id and .sender = sender.id and .product = product),
  inserted_loan := (insert Loan {
      amount := <int64>$amount,
      receiver := receiver.id,
      sender := sender.id,
      product := product,
      date := <datetime>$date
  }),
  loan := exist_loan ?? inserted_loan
update Bank set {
  loans += loan,
  money := <int64>$bank_money
};
update User filter .userid = <int64>$receiver_id set {money := <int64>$receiver_money};