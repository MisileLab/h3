update Loan filter .id = <uuid>$id set {
  date := <datetime>$date,
  interest := <int64>$interest
};
update User filter .id = <uuid>$userid set {
  credit := <int64>$credit
};
