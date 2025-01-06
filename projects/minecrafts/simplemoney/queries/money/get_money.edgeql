insert User {
  userid := <int64>$userid
} unless conflict on .userid;
select (select User {credit} filter .userid = <int64>$userid).credit;

