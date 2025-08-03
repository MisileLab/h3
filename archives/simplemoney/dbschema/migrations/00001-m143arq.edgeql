CREATE MIGRATION m143arqadajc3y5h5krsoeh4kcyonejuzcig52pqbpz66hgitehn3a
    ONTO initial
{
  CREATE TYPE default::Data {
      CREATE REQUIRED PROPERTY amount: std::int64 {
          CREATE CONSTRAINT std::min_value(0);
      };
      CREATE REQUIRED PROPERTY receiver: std::uuid;
      CREATE REQUIRED PROPERTY sender: std::uuid;
  };
  CREATE TYPE default::Product {
      CREATE REQUIRED PROPERTY end_date: std::int64 {
          CREATE CONSTRAINT std::min_value(0);
      };
      CREATE REQUIRED PROPERTY interest: std::int64 {
          CREATE CONSTRAINT std::min_value(0);
      };
      CREATE REQUIRED PROPERTY min_trust: std::int64;
      CREATE REQUIRED PROPERTY name: std::str {
          CREATE CONSTRAINT std::exclusive;
      };
  };
  CREATE TYPE default::Loan EXTENDING default::Data {
      CREATE REQUIRED LINK product: default::Product;
      CREATE REQUIRED PROPERTY date: std::datetime;
      CREATE REQUIRED PROPERTY interest: std::int64 {
          CREATE CONSTRAINT std::min_value(0);
      };
  };
  CREATE TYPE default::User {
      CREATE MULTI LINK banks: default::Data;
      CREATE MULTI LINK transactions: default::Data;
      CREATE MULTI LINK loans: default::Loan {
          ON TARGET DELETE ALLOW;
      };
      CREATE REQUIRED PROPERTY credit: std::int64 {
          SET default := 0;
          CREATE CONSTRAINT std::min_value(0);
      };
      CREATE REQUIRED PROPERTY money: std::int64 {
          SET default := 0;
          CREATE CONSTRAINT std::exclusive;
          CREATE CONSTRAINT std::min_value(0);
      };
      CREATE REQUIRED PROPERTY userid: std::int64 {
          CREATE CONSTRAINT std::exclusive;
          CREATE CONSTRAINT std::min_value(0);
      };
  };
  CREATE TYPE default::Bank {
      CREATE MULTI LINK loans: default::Loan {
          ON TARGET DELETE ALLOW;
      };
      CREATE REQUIRED LINK owner: default::User;
      CREATE MULTI LINK products: default::Product {
          ON TARGET DELETE ALLOW;
      };
      CREATE MULTI LINK transactions: default::Data;
      CREATE REQUIRED PROPERTY money: std::int64 {
          SET default := 0;
          CREATE CONSTRAINT std::min_value(0);
      };
      CREATE REQUIRED PROPERTY name: std::str {
          CREATE CONSTRAINT std::exclusive;
      };
  };
};
