CREATE MIGRATION m17qeo6gyqsa34hlvaixzvblxo4lxw3nmekdahe2xwy4nwr5zvmtoq
    ONTO initial
{
  CREATE TYPE default::Account {
      CREATE REQUIRED PROPERTY money: std::int64;
      CREATE REQUIRED PROPERTY name: std::str {
          CREATE CONSTRAINT std::exclusive;
      };
      CREATE REQUIRED PROPERTY password: std::str;
  };
  CREATE TYPE default::Transaction {
      CREATE REQUIRED LINK received: default::Account;
      CREATE REQUIRED LINK sent: default::Account;
      CREATE REQUIRED PROPERTY amount: std::int64;
  };
  ALTER TYPE default::Account {
      CREATE MULTI LINK transactions: default::Transaction;
  };
};
