CREATE MIGRATION m16ezz7au6airrzrudquofk3qfpqb3otmffphdhig7aeklt5sogleq
    ONTO m1cfbtatd5jtqmw6hu76drx5smb5hpzprjq4srnrcnl64vxuw3fnfa
{
  ALTER TYPE default::Bank {
      ALTER LINK loans {
          SET OWNED;
      };
      ALTER LINK transactions {
          SET OWNED;
      };
  };
  ALTER TYPE default::User {
      ALTER PROPERTY credit {
          ALTER CONSTRAINT std::min_value(0) {
              SET OWNED;
          };
          SET OWNED;
      };
      ALTER PROPERTY money {
          ALTER CONSTRAINT std::exclusive {
              SET OWNED;
          };
          ALTER CONSTRAINT std::min_value(0) {
              SET OWNED;
          };
          SET OWNED;
      };
      ALTER LINK loans {
          SET OWNED;
      };
      ALTER LINK transactions {
          SET OWNED;
      };
      DROP EXTENDING default::BaseUser;
  };
  ALTER TYPE default::BaseUser {
      DROP PROPERTY credit;
      DROP PROPERTY money;
      DROP LINK loans;
      DROP LINK transactions;
  };
  ALTER TYPE default::Bank {
      DROP EXTENDING default::BaseUser;
      ALTER LINK loans {
          ON TARGET DELETE ALLOW;
          RESET readonly;
          SET MULTI;
          SET TYPE default::Loan;
      };
      ALTER LINK transactions {
          RESET readonly;
          SET MULTI;
          SET TYPE default::Data;
      };
  };
  ALTER TYPE default::User {
      ALTER LINK loans {
          ON TARGET DELETE ALLOW;
          RESET readonly;
          SET MULTI;
          SET TYPE default::Loan;
      };
      ALTER LINK transactions {
          RESET readonly;
          SET MULTI;
          SET TYPE default::Data;
      };
      ALTER PROPERTY credit {
          SET default := 0;
          RESET readonly;
          RESET CARDINALITY;
          SET REQUIRED;
          SET TYPE std::int64;
      };
      ALTER PROPERTY money {
          SET default := 0;
          RESET readonly;
          RESET CARDINALITY;
          SET REQUIRED;
          SET TYPE std::int64;
      };
  };
  ALTER TYPE default::Bank {
      CREATE REQUIRED PROPERTY amount: std::int64 {
          SET default := 0;
          CREATE CONSTRAINT std::min_value(0);
      };
  };
  DROP TYPE default::BaseUser;
};
