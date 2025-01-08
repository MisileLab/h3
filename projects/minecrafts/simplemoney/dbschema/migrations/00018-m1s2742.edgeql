CREATE MIGRATION m1s2742bkzxsjqvnwzojok2533blsey3fb3ukgr5iu6irobhnyfcaa
    ONTO m16ozqqdr6jextwf2z7akvtvxadzq5apadrushkjk2mnzuunxfbq3a
{
  ALTER TYPE default::Product {
      CREATE REQUIRED LINK bank: default::Bank {
          SET REQUIRED USING (SELECT
              default::Bank 
          LIMIT
              1
          );
      };
  };
};
