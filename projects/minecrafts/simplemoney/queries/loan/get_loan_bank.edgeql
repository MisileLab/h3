select (select Bank {loans: {date, amount, receiver, product: {interest, name}}} filter .name = <str>$name).loans;
