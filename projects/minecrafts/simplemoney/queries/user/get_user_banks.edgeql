select (select User { banks: {amount, sender}, money } filter .userid = <int64>$userid);

