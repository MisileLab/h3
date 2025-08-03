CREATE MIGRATION m1qkrr7rx6sm7j6dvrylvn36xbwazsdx2rxrepuokwjmztvijktwtq
    ONTO m1eseay2txbjwcippem55c7i4v3qnz46fwunxdf3dervlbhmdvqruq
{
  ALTER TYPE default::User {
      ALTER LINK banks {
          ON TARGET DELETE ALLOW;
      };
  };
};
