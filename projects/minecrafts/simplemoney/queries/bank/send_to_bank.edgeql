with
  receiver := (select Bank filter .id = <uuid>$receiver),
  sender := (select Bank filter .id = <uuid>$sender),
  data := (insert Data {amount := <int64>$amount, sender := sender.id, receiver := receiver.id}),
  def := (update sender set {transactions += data, money := <int64>$sender_money})
update receiver set {transactions += data, money := <int64>$receiver_money};

