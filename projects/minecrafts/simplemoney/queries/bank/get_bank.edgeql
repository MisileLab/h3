select default::Bank {
  id,
  amount,
  name,
  owner: {userid}
}
filter .name = <str>$name
