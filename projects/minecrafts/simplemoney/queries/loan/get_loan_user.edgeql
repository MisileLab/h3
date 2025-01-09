with
  user := (select User {id} filter .userid = <int64>$userid)
select Loan {id, receiver, date, product: {end_date, interest, name}, amount} filter .receiver = user.id;
