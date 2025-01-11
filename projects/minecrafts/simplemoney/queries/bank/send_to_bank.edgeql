with
  receiver := (select Bank filter .id = <uuid>$receiver),
  sender := (select Bank filter .id = <uuid>$sender),
  data := (insert Data {amount := <int64>$amount, sender := sender.id, receiver := receiver.id}),
  def := (update sender set {transactions += data, money := .money - <int64>$amount}),
update receiver set {
  transactions += data,
  money := .money + <int64>$amount - <int64>math::ceil(<int64>$amount / 100 * <int64>$fee)
};

