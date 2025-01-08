select Loan {id, receiver, date, product: {end_date, interest}, amount, interest} filter .date < datetime_current();
