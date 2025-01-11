select default::Bank {
  id,
  money,
  owner: {userid}
}
filter .name = <str>$name
