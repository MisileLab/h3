CREATE MIGRATION m1gt7t4nj7m5wkwgdr57yt25ykrqqpwb4j5l6kzcpm243vyf5ohzvq
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
      CREATE REQUIRED PROPERTY interest: std::float64 {
          CREATE CONSTRAINT std::min_value(0);
      };
      CREATE REQUIRED PROPERTY min_trust: std::int64;
      CREATE REQUIRED PROPERTY name: std::str {
          CREATE CONSTRAINT std::exclusive;
      };
  };
  CREATE TYPE default::Bank {
      CREATE MULTI LINK borrows: default::Data;
      CREATE MULTI LINK products: default::Product;
      CREATE MULTI LINK transactions: default::Data;
      CREATE PROPERTY amount: std::int64;
  };
  CREATE TYPE default::User {
      CREATE MULTI LINK banks: default::Bank;
      CREATE MULTI LINK transactions: default::Data;
      CREATE PROPERTY credit: std::int64 {
          CREATE CONSTRAINT std::min_value(0);
      };
      CREATE REQUIRED PROPERTY name: std::str {
          CREATE CONSTRAINT std::exclusive;
      };
      CREATE PROPERTY trust: std::int64;
      CREATE REQUIRED PROPERTY userid: std::int64 {
          CREATE CONSTRAINT std::exclusive;
          CREATE CONSTRAINT std::min_value(0);
      };
  };
  ALTER TYPE default::Bank {
      CREATE MULTI LINK owners: default::User;
  };
  CREATE TYPE default::Borrow EXTENDING default::Data, default::Product {
      CREATE CONSTRAINT std::exclusive ON ((.receiver, .sender, .name));
      CREATE REQUIRED PROPERTY product_value := (.end_date);
  };
};
