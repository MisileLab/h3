CREATE MIGRATION m1lzxbk473aiunen4u3rgnpd7a3iwpgtrg6evyehug6bpj2ifsjwfq
    ONTO m1oslbofjbnepmat3cutqqm3kaosegn5mz6ju447znituurmwicimq
{
  ALTER TYPE default::Bank {
      DROP PROPERTY amount;
  };
  ALTER TYPE default::Bank {
      CREATE PROPERTY amount: std::int64 {
          SET default := 0;
          CREATE CONSTRAINT std::exclusive;
          CREATE CONSTRAINT std::min_value(0);
      };
  };
  ALTER TYPE default::User {
      DROP PROPERTY credit;
  };
  ALTER TYPE default::User {
      CREATE PROPERTY credit: std::int64 {
          SET default := 0;
          CREATE CONSTRAINT std::exclusive;
          CREATE CONSTRAINT std::min_value(0);
      };
  };
  ALTER TYPE default::User {
      DROP PROPERTY trust;
  };
  ALTER TYPE default::User {
      CREATE PROPERTY trust: std::int64 {
          SET default := 0;
          CREATE CONSTRAINT std::exclusive;
          CREATE CONSTRAINT std::min_value(0);
      };
  };
};
