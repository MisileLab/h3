CREATE MIGRATION m1pdodlll7cy4fzjjfld66s4ei6onpkqpqrpwkuofpialkvwr7py5a
    ONTO initial
{
  CREATE TYPE default::User {
      CREATE PROPERTY aboutme: std::str;
      CREATE PROPERTY name: std::str;
      CREATE PROPERTY pnumber: std::str;
      CREATE PROPERTY portfolio: std::str;
      CREATE PROPERTY why: std::str;
  };
};
