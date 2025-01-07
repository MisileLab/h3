CREATE MIGRATION m1rrumwouvoxcwetqz6d43t6gl3wzusfefejq2v74zeeg44ffqnt2q
    ONTO m1yrg4gjuugc6vqekwjch6tr5nywjv4kgyx46dpbuynitxctceflxa
{
  ALTER TYPE default::Bank {
      ALTER PROPERTY name {
          CREATE CONSTRAINT std::exclusive;
      };
  };
};
