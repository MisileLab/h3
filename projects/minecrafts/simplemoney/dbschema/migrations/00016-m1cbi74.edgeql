CREATE MIGRATION m1cbi74zwvrxd7yomxsmg4czqcf7ehc6m4jlqi75dwkbw3zpqawpqa
    ONTO m1c3yce4nbpzvqruihqxjisukeqne3quyothifglet3dkfileli3dq
{
  ALTER TYPE default::Bank {
      DROP LINK borrows;
  };
  ALTER TYPE default::Borrow RENAME TO default::Loan;
  ALTER TYPE default::Bank {
      CREATE MULTI LINK loans: default::Loan {
          ON TARGET DELETE ALLOW;
      };
      ALTER LINK products {
          ON TARGET DELETE ALLOW;
      };
  };
  ALTER TYPE default::User {
      ALTER LINK borrows {
          RENAME TO loans;
      };
  };
};
