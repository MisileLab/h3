delete (select Bank {products: {name}} filter .name = <str>$bank_name and .products.name = <str>$name)
