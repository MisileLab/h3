CREATE MIGRATION m1tim4i5w5oxb6ap3m5yknshzsspivcw72oolylfcixhdwktpcrd5a
    ONTO m12brxksi3i6ii5joltkse2x5d3rp6r6lwxoiyc4uy7ewsqgnxs7aa
{
  ALTER TYPE default::Bank {
      CREATE REQUIRED LINK owner: default::User {
          SET REQUIRED USING (<default::User>{});
      };
  };
  ALTER TYPE default::Bank {
      DROP LINK owners;
  };
};
