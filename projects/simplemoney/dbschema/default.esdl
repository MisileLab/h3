module default {
  type Account {
    required money: int64;
    required name: str { constraint exclusive };
    required password: str;
    multi transactions: Transaction;
  }

  type Transaction {
    required sent: Account;
    required received: Account;
    required amount: int64;
  }
}
