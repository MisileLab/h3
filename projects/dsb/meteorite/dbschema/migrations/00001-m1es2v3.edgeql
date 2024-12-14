CREATE MIGRATION m1es2v37qb7nftvrnmm2pbehvmf54jyxr3psluw5umgqmoustr7ysa
    ONTO initial
{
  CREATE MODULE Lunch IF NOT EXISTS;
  CREATE TYPE Lunch::School {
      CREATE REQUIRED PROPERTY name: std::str;
      CREATE REQUIRED PROPERTY ofcdc_code: std::str;
      CREATE REQUIRED PROPERTY school_code: std::int64;
  };
};
