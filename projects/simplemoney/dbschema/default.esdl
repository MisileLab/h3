module default {
  type Account {
    required money: int64;
    required name: str { constraint exclusive };
    required password: str;
    multi transactions: Transaction;
  }

  type Transaction {
    to: str;
    received: str;
    required amount: int64;
  }
}
