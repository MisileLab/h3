delete Loan filter .receiver = (select User filter .userid = <int64>$userid).id;
