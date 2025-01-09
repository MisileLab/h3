with
  products := (select Bank {products: {interest, end_date}} filter .name = <str>$bank_name).products
select products {interest, end_date} filter .name = <str>$name;