CREATE MIGRATION m1cfbtatd5jtqmw6hu76drx5smb5hpzprjq4srnrcnl64vxuw3fnfa
    ONTO m1ytl37z7gzmfpfllatt4ml3p5h4wzhqulrs4syhqluwqt5qsbxwcq
{
  ALTER TYPE default::Data {
      DROP LINK receiver;
  };
  ALTER TYPE default::Data {
      DROP LINK sender;
  };
  ALTER TYPE default::Data {
      CREATE REQUIRED PROPERTY receiver: std::uuid {
          SET REQUIRED USING (<std::uuid>{});
      };
  };
  ALTER TYPE default::Data {
      CREATE REQUIRED PROPERTY sender: std::uuid {
          SET REQUIRED USING (<std::uuid>{});
      };
  };
};
