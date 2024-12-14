CREATE MIGRATION m1f4fybdqj6riqmuwekoiervwdpgaouo7ouxo4vo47tusspew44u7a
    ONTO m1es2v37qb7nftvrnmm2pbehvmf54jyxr3psluw5umgqmoustr7ysa
{
  ALTER TYPE Lunch::School {
      ALTER PROPERTY name {
          CREATE CONSTRAINT std::exclusive;
      };
      ALTER PROPERTY ofcdc_code {
          CREATE CONSTRAINT std::exclusive;
      };
      ALTER PROPERTY school_code {
          CREATE CONSTRAINT std::exclusive;
      };
  };
};
