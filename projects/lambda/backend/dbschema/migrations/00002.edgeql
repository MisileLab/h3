CREATE MIGRATION m1bqumzxw3x3hwn73cnn6h55jwvrmuuli75udavcrn2syrhvnxeulq
    ONTO m1pdodlll7cy4fzjjfld66s4ei6onpkqpqrpwkuofpialkvwr7py5a
{
  ALTER TYPE default::User {
      ALTER PROPERTY aboutme {
          RENAME TO me;
      };
  };
  ALTER TYPE default::User {
      ALTER PROPERTY me {
          SET REQUIRED USING (<std::str>{});
      };
      ALTER PROPERTY name {
          SET REQUIRED USING (<std::str>{});
      };
      ALTER PROPERTY pnumber {
          SET REQUIRED USING (<std::str>{});
      };
      ALTER PROPERTY why {
          SET REQUIRED USING (<std::str>{});
      };
  };
};
