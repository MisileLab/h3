insert User {
  credit := 0,
  userid := <int64>$userid
} unless conflict on .userid;
select (select User { banks: {amount, sender}, money } filter .userid = <int64>$userid);

