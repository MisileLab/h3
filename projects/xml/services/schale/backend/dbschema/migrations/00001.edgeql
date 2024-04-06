CREATE MIGRATION m1poc2vesummkffhasokumykgdd54kbnzhu5i64blbbnvt5xmdh45q
    ONTO initial
{
  CREATE TYPE default::User {
      CREATE REQUIRED PROPERTY pw: std::str;
      CREATE REQUIRED PROPERTY userid: std::str;
  };
};
