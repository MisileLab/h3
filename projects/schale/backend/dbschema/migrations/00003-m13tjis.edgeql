CREATE MIGRATION m13tjisyuvrhneiyghynl2vg2grf5uixcgpdpwh4zycmego4yhdrea
    ONTO m1svkjtotn3wsew5izlcu5mqyt6uuwz5s3wvjphmmkiobunrvomlhq
{
  CREATE MODULE theresa IF NOT EXISTS;
  ALTER TYPE default::Letter RENAME TO theresa::Letter;
  ALTER TYPE default::User RENAME TO theresa::User;
};
