CREATE MIGRATION m1bb5ulhzdcsfyzti4t4ufhxjfu4radkbxedalcgx6f7b2vr63xeoq
    ONTO m1qvjft2w2ov6yzx7o6pe7yzvbvd5lmcwbfkrj4bmk6c2sbrtq2zcq
{
  ALTER TYPE default::Bank {
      DROP PROPERTY amount;
  };
  CREATE TYPE default::BaseUser {
      CREATE MULTI LINK loans: default::Loan {
          ON TARGET DELETE ALLOW;
      };
      CREATE MULTI LINK transactions: default::Data;
      CREATE REQUIRED PROPERTY money: std::int64 {
          SET default := 0;
          CREATE CONSTRAINT std::exclusive;
          CREATE CONSTRAINT std::min_value(0);
      };
  };
  ALTER TYPE default::Bank {
      EXTENDING default::BaseUser LAST;
      ALTER LINK loans {
          RESET CARDINALITY;
          DROP OWNED;
          RESET TYPE;
      };
      ALTER LINK transactions {
          RESET CARDINALITY;
          DROP OWNED;
          RESET TYPE;
      };
  };
  ALTER TYPE default::User EXTENDING default::BaseUser LAST;
  ALTER TYPE default::User {
      ALTER LINK loans {
          RESET CARDINALITY;
          DROP OWNED;
          RESET TYPE;
      };
      ALTER LINK transactions {
          RESET CARDINALITY;
          DROP OWNED;
          RESET TYPE;
      };
      ALTER PROPERTY money {
          ALTER CONSTRAINT std::exclusive {
              DROP OWNED;
          };
          RESET OPTIONALITY;
          ALTER CONSTRAINT std::min_value(0) {
              DROP OWNED;
          };
          DROP OWNED;
          RESET TYPE;
      };
  };
  ALTER TYPE default::Data {
      DROP PROPERTY receiver;
  };
  ALTER TYPE default::Data {
      CREATE REQUIRED LINK receiver: default::BaseUser {
          SET REQUIRED USING (<default::BaseUser>{});
      };
  };
  ALTER TYPE default::Data {
      DROP PROPERTY sender;
  };
  ALTER TYPE default::Data {
      CREATE REQUIRED LINK sender: default::BaseUser {
          SET REQUIRED USING (<default::BaseUser>{});
      };
  };
};
