with
  user := (select User {id} filter .userid = <int64>$userid)
select Loan {amount}
filter .receiver = user.id and .sender = <uuid>$sender and .product.name = <str>$product_name
limit 1;
