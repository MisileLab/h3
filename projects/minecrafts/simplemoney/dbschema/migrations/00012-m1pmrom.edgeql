CREATE MIGRATION m1pmrom24frr3475joflqbocppwvz4mykg7kvzz4jkmoklgrq374za
    ONTO m1tim4i5w5oxb6ap3m5yknshzsspivcw72oolylfcixhdwktpcrd5a
{
  ALTER TYPE default::Bank {
      ALTER PROPERTY amount {
          SET REQUIRED USING (<std::int64>{});
      };
  };
  ALTER TYPE default::User {
      ALTER PROPERTY credit {
          SET REQUIRED USING (<std::int64>{});
      };
      ALTER PROPERTY trust {
          SET REQUIRED USING (<std::int64>{});
      };
  };
};
