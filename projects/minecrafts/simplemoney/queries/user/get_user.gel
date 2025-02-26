insert User {
  credit := 0,
  userid := <int64>$userid
} unless conflict on .userid;
select User {credit, userid} filter .id = <uuid>$id;
