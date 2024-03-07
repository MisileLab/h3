CREATE MIGRATION m1bp4peg3wiazat6wad3lgrhernlkcgolrnlhwitgs443tas3yew2a
    ONTO m12y4yy5lppcz5xavjqg2uyxlzryfmuos6b5h6u2g5gmdwhsg4n4ra
{
  ALTER TYPE default::User {
      CREATE INDEX ON (.name);
      ALTER PROPERTY time {
          RESET OPTIONALITY;
      };
  };
};
