module default {
  type User {
    required name: str {
      constraint exclusive;
    };
    required userid: uint64 {
      constraint exclusive;
    };
    trust: int64;
    credit: uint64;
    multi transactions: Data;
    multi banks: Bank;
  }

  type Borrow extending Data, Product {
    required product_value := (.end_date);
    constraint exclusive on ((.receiver, .sender, .name));
  }

  type Data {
    required amount: uint64;
    required receiver: uuid;
    required sender: uuid;
  }

  type Bank {
    amount: int64;
    multi products: Product;
    multi borrows: Data;
    multi transactions: Data;
    multi owners: User;
  }

  type Product {
    required name: str {
      constraint exclusive;
    };
    required interest: float64 {
      constraint min_value(0);
    };
    required end_date: uint64;
    required min_trust: int64;
  }
}
