CREATE MIGRATION m1svkjtotn3wsew5izlcu5mqyt6uuwz5s3wvjphmmkiobunrvomlhq
    ONTO m14gsranxohefkxfhebt2ke2axeswavhkfbc4dirnsocdpb4k3t7sq
{
  ALTER TYPE default::User {
      ALTER PROPERTY email {
          DROP CONSTRAINT std::exclusive;
      };
      CREATE REQUIRED PROPERTY hash: std::str {
          SET REQUIRED USING (<std::str>{});
          CREATE CONSTRAINT std::exclusive;
      };
  };
};
