module default {
  type User {
    required userid: int64 {
      constraint exclusive;
      constraint min_value(0);
    };
    required credit: int64 {
      constraint min_value(0);
      default := 0;
    };
    required money: int64 {
      constraint exclusive;
      constraint min_value(0);
      default := 0;
    };
    multi transactions: Data;
    multi banks: Bank;
    multi loans: Loan {
      on target delete allow;
    };
  }

  type Loan extending Data {
    required product: Product;
    required date: datetime;
    required interest: int64 {
      constraint min_value(0);
    };
  }

  type Data {
    required amount: int64 {
      constraint min_value(0);
    };
    required receiver: uuid;
    required sender: uuid;
  }

  type Bank {
    required name: str {
      constraint exclusive;
    };
    required money: int64 {
      constraint min_value(0);
      default := 0;
    };
    multi products: Product {
      on target delete allow;
    };
    multi loans: Loan {
      on target delete allow;
    };
    multi transactions: Data;
    required owner: User;
  }

  type Product {
    required name: str {
      constraint exclusive;
    };
    required interest: int64 {
      constraint min_value(0);
    };
    required end_date: int64 {
      constraint min_value(0);
    };
    required min_trust: int64;
  }
}
