CREATE MIGRATION m1eseay2txbjwcippem55c7i4v3qnz46fwunxdf3dervlbhmdvqruq
    ONTO m1lg2thjuecgxqq66ju7nzavzbhdi4cln52asrz3dwffsb5vgcrdba
{
  ALTER TYPE default::Bank {
      DROP LINK incomes;
  };
};
