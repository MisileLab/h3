with
  receiver := (select User filter .userid = <int64>$receiver_id),
  sender := (select Bank filter .name = <str>$bank_name),
  product := (select Product filter .id = <uuid>$product_id),
  exist_loan := (
    update Loan filter .receiver = receiver.id and .sender = sender.id and .product = product set {
      amount := .amount + <int64>$amount
    }
  ),
  loan := exist_loan ?? (insert Loan {
    amount := <int64>$amount,
    receiver := receiver.id,
    sender := sender.id,
    product := product,
    date := <datetime>$date
  }),
  def := exist_loan ?? (update sender set {
    loans += loan
  }).loans,
  def2 := exist_loan ?? (update receiver set {
    loans += loan
  }).loans
update sender set {
  money := <int64>$bank_money
};
update User filter .userid = <int64>$receiver_id set {money := <int64>$receiver_money};