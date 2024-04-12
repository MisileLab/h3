CREATE MIGRATION m1xyycopocle62c4zt55rxjub7fjzcj4fpcpbeo5t4rcziylmjuuhq
    ONTO m1mvzjfkikf4frqwbohixzqxxcyanv2odywn3yiv6ez2wykxwnez3q
{
  ALTER TYPE default::User {
      CREATE PROPERTY admin: std::bool;
      CREATE PROPERTY userid: std::str;
  };
};
