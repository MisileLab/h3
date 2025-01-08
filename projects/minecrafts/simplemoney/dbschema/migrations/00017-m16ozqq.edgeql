CREATE MIGRATION m16ozqqdr6jextwf2z7akvtvxadzq5apadrushkjk2mnzuunxfbq3a
    ONTO m1cbi74zwvrxd7yomxsmg4czqcf7ehc6m4jlqi75dwkbw3zpqawpqa
{
  ALTER TYPE default::User {
      ALTER PROPERTY credit {
          DROP CONSTRAINT std::min_value(0);
      };
  };
};
