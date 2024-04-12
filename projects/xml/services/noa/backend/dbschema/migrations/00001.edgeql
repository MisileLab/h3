CREATE MIGRATION m1mvzjfkikf4frqwbohixzqxxcyanv2odywn3yiv6ez2wykxwnez3q
    ONTO initial
{
  CREATE TYPE default::KeyStore {
      CREATE PROPERTY name: std::str;
      CREATE PROPERTY privkey: std::str;
      CREATE PROPERTY pubkey: std::str;
  };
  CREATE TYPE default::User {
      CREATE MULTI LINK groups: default::KeyStore {
          CREATE PROPERTY name: std::str;
          CREATE PROPERTY pubkey: std::str;
      };
  };
};
