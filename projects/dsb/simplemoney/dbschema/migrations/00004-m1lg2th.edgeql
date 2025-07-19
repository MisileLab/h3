CREATE MIGRATION m1lg2thjuecgxqq66ju7nzavzbhdi4cln52asrz3dwffsb5vgcrdba
    ONTO m1wxmsdi56otftz6m5wiflafrphmekrl5ownd2vei35pae3vp4xnxa
{
  ALTER TYPE default::Bank {
      CREATE MULTI LINK incomes: default::Data {
          ON TARGET DELETE ALLOW;
      };
  };
};
