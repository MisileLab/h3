CREATE MIGRATION m1ppoyocfo7essetm7dg6s2xk7td4xkx5fkzzzck3ueaaoctqgvnza
    ONTO m1wsiqdippiutcgkw422gdt7iuxxxlhy7nf2h6t7dgmeghd22sp4ba
{
  ALTER TYPE default::User {
      DROP CONSTRAINT std::exclusive ON (.name);
  };
};
