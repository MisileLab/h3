select (select Bank {loans: {date, interest, amount, receiver, product: {interest}}} filter .name = <str>$name).loans;
