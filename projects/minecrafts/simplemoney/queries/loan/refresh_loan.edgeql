update Loan filter .id = <uuid>$id set {
  date := <datetime>$date,
  amount := <int64>$amount
};
update User filter .id = <uuid>$userid set {
  credit := <int64>$credit
};
