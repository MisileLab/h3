CREATE MIGRATION m1c3yce4nbpzvqruihqxjisukeqne3quyothifglet3dkfileli3dq
    ONTO m1dz457az5wtplbaj2hyel4x2jjss4f5dvr6mzbacxnwyzc5bkjpgq
{
  ALTER TYPE default::User {
      ALTER LINK borrows {
          ON TARGET DELETE ALLOW;
      };
  };
};
