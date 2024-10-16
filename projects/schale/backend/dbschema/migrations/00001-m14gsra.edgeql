CREATE MIGRATION m14gsranxohefkxfhebt2ke2axeswavhkfbc4dirnsocdpb4k3t7sq
    ONTO initial
{
  CREATE TYPE default::User {
      CREATE REQUIRED PROPERTY email: std::str {
          CREATE CONSTRAINT std::exclusive;
      };
      CREATE REQUIRED PROPERTY message: std::str;
      CREATE REQUIRED PROPERTY name: std::str;
      CREATE PROPERTY signature: std::bytes;
  };
  CREATE TYPE default::Letter {
      CREATE MULTI LINK signers: default::User;
      CREATE REQUIRED PROPERTY name: std::str {
          CREATE CONSTRAINT std::exclusive;
      };
      CREATE REQUIRED PROPERTY tldr: std::str;
  };
};
