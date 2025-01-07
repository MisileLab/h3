CREATE MIGRATION m1dz457az5wtplbaj2hyel4x2jjss4f5dvr6mzbacxnwyzc5bkjpgq
    ONTO m1tapc3fscdlpmgh4tf7mnekesqohlgbilyetbaa66glmcosfcj2bq
{
  ALTER TYPE default::Bank {
      ALTER LINK borrows {
          SET TYPE default::Borrow USING (.borrows[IS default::Borrow]);
      };
  };
  ALTER TYPE default::User {
      CREATE MULTI LINK borrows: default::Borrow;
  };
};
