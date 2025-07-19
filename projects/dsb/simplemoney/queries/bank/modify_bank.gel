with
  owner := (select User filter .userid = <int64>$owner)
insert Bank {
  name := <str>$name,
  money := <int64>$money,
  owner := owner
} unless conflict on .name else (
  update Bank set {
  money := <int64>$money,
  owner := owner
})
