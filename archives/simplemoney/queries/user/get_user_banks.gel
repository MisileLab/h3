insert User {
  credit := 0,
  userid := <int64>$userid
} unless conflict on .userid;
select (select User { banks: {amount, receiver}, money, credit } filter .userid = <int64>$userid);

