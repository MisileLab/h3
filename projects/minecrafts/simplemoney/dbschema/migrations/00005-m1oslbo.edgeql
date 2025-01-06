CREATE MIGRATION m1oslbofjbnepmat3cutqqm3kaosegn5mz6ju447znituurmwicimq
    ONTO m15nfokkoayn73hr756lkgzxyxscwbxhzpzsaoybuwffvf6nnmxsza
{
  ALTER TYPE default::Bank {
      ALTER PROPERTY amount {
          USING (0);
      };
  };
  ALTER TYPE default::User {
      ALTER PROPERTY credit {
          USING (0);
          DROP CONSTRAINT std::min_value(0);
      };
      ALTER PROPERTY trust {
          USING (0);
      };
  };
};
