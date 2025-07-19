CREATE MIGRATION m1ox6iew7ulje4csa6qqijv7xktjq3qb2hpehhodnnisaiphpjg5ua
    ONTO m143arqadajc3y5h5krsoeh4kcyonejuzcig52pqbpz66hgitehn3a
{
  ALTER TYPE default::Loan {
      DROP PROPERTY interest;
  };
};
