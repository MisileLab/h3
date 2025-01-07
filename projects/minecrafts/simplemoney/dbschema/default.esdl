module default {
  type User {
    required userid: int64 {
      constraint exclusive;
      constraint min_value(0);
    };
    required credit: int64 {
      constraint exclusive;
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
    multi borrows: Borrow {
      on target delete allow;
    };
  }

  type Borrow extending Data {
    required product: Product;
    required date: int64 {
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
    required amount: int64 {
      constraint min_value(0);
      default := 0;
    };
    multi products: Product;
    multi borrows: Borrow;
    multi transactions: Data;
    required owner: User;
  }

  type Product {
    required name: str {
      constraint exclusive;
    };
    required interest: float64 {
      constraint min_value(0);
    };
    required end_date: int64 {
      constraint min_value(0);
    };
    required min_trust: int64;
  }
}
