with
  receiver := (select User filter .userid = <int64>$receiver_id),
  sender := (select Bank filter .name = <str>$bank_name),
  product := (select Product filter .id = <uuid>$product_id),
  computed_fee := (<int64>math::ceil(<int64>$amount / 100 * <int64>$fee)),
  exist_loan := (
    update Loan filter .receiver = receiver.id and .sender = sender.id and .product = product set {
      amount := <int64>$amount + computed_fee
    }
  ),
  loan := exist_loan ?? (insert Loan {
    amount := <int64>$amount + computed_fee,
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
  }).loans,
  def3 := (update sender set {
    money := .money - <int64>$amount
  }),
  def4 := (update receiver set {
    money := .money + <int64>$amount - computed_fee
  })
select {def, def2, def3, def4};