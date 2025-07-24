CREATE MIGRATION m1dxjwitn2lxz5zzbnzsgi2besipabbg7dwovoyzlso5mibsnrepxq
    ONTO initial
{
  CREATE FUTURE simple_scoping;
  CREATE TYPE default::Data {
      CREATE PROPERTY author_name: std::str;
      CREATE PROPERTY content: std::str;
      CREATE CONSTRAINT std::exclusive ON ((.author_name, .content));
      CREATE PROPERTY is_bot: std::bool;
  };
};
