CREATE MIGRATION m1yw5cr2t3s4dprwgdvt5fus22izda4w7cpxgmb3gji3nsckekbcda
    ONTO m1bfipg4oixb2kbflckgik24zcvtzyfgkhvr7wnnsiodhmgfzf7sea
{
  ALTER TYPE default::Transaction {
      ALTER PROPERTY received {
          RESET CARDINALITY USING (SELECT
              .received 
          LIMIT
              1
          );
      };
      ALTER PROPERTY sent {
          RESET CARDINALITY USING (SELECT
              .sent 
          LIMIT
              1
          );
      };
  };
};
