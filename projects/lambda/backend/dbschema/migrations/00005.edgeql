CREATE MIGRATION m1wsiqdippiutcgkw422gdt7iuxxxlhy7nf2h6t7dgmeghd22sp4ba
    ONTO m1bp4peg3wiazat6wad3lgrhernlkcgolrnlhwitgs443tas3yew2a
{
  ALTER TYPE default::User {
      CREATE CONSTRAINT std::exclusive ON (.name);
      DROP INDEX ON (.name);
      ALTER PROPERTY time {
          SET REQUIRED USING (<std::float64>{});
      };
  };
};
