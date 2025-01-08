CREATE MIGRATION m1ir233j3nmz7mgiwbtpl45x54sveujb3c6pxzvej6p3vqua2sftma
    ONTO m16ezz7au6airrzrudquofk3qfpqb3otmffphdhig7aeklt5sogleq
{
  ALTER TYPE default::Bank {
      ALTER PROPERTY amount {
          RENAME TO money;
      };
  };
};
