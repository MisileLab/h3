with
  receiver := (select Bank filter .id = <uuid>$receiver),
  sender := (select User {banks} filter .userid = <int64>$senderid),
  banks := (select sender.banks {id, receiver, amount} filter .receiver = <uuid>$receiver),
  data := (insert Data {amount := <int64>$amount, sender := sender.id, receiver := receiver.id}),
  bank := (
    (update User filter .userid = <int64>$senderid set {banks += (insert Data {
      amount := <int64>$amount,
      sender := sender.id,
      receiver := <uuid>$receiver
    })}).banks if not exists banks else (update banks set {
      amount := .amount + <int64>$amount
    })
  ),
  def := (update sender set {transactions += data, money := .money - <int64>$amount})
update receiver set {transactions += data, money := .money + <int64>$amount};
