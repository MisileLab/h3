CREATE MIGRATION m1tapc3fscdlpmgh4tf7mnekesqohlgbilyetbaa66glmcosfcj2bq
    ONTO m1pmrom24frr3475joflqbocppwvz4mykg7kvzz4jkmoklgrq374za
{
  ALTER TYPE default::User {
      ALTER PROPERTY trust {
          RENAME TO money;
      };
  };
};
