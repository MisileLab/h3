with
  owner := (select User filter .userid = <int64>$owner)
insert Bank {
  name := <str>$name,
  amount := <int64>$amount,
  owner := owner
} unless conflict on .name else (
  update Bank set {
  amount := <int64>$amount,
  owner := owner
})
