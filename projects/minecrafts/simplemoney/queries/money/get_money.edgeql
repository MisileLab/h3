insert User {
  userid := <int64>$userid
} unless conflict on .userid;
select (select User {money} filter .userid = <int64>$userid).money;

