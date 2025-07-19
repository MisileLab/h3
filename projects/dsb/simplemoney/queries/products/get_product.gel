with
  products := (select Bank {products: {interest, end_date, max_amount, min_trust}} filter .name = <str>$bank_name).products
select products {interest, end_date, max_amount, min_trust} filter .name = <str>$name;