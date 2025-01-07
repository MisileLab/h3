insert User {
  userid := <int64>$userid,
  credit := <int64>$credit
} unless conflict on .userid else (
  update User set {
    credit := <int64>$credit
  })
