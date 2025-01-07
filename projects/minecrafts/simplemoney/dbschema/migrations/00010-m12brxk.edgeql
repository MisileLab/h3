CREATE MIGRATION m12brxksi3i6ii5joltkse2x5d3rp6r6lwxoiyc4uy7ewsqgnxs7aa
    ONTO m1rrumwouvoxcwetqz6d43t6gl3wzusfefejq2v74zeeg44ffqnt2q
{
  ALTER TYPE default::Bank {
      ALTER PROPERTY amount {
          DROP CONSTRAINT std::exclusive;
      };
  };
};
