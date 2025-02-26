with
  receiver := (select User filter .userid = <int64>$receiverid),
  sender := (select User filter .userid = <int64>$senderid),
  data := (insert Data {amount := <int64>$amount, sender := sender.id, receiver := receiver.id}),
  def := (update sender set {transactions += data, money := .money - <int64>$amount}),
  def2 := (update receiver set {
    transactions += data,
    money := .money + <int64>$amount - <int64>(math::ceil(<int64>$amount / 100 * <float64>$fee))
  })
select {def, def2};
