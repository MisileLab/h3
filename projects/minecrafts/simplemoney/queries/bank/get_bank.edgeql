select default::Bank {
  id,
  money,
  name,
  owner: {userid}
}
filter .name = <str>$name
