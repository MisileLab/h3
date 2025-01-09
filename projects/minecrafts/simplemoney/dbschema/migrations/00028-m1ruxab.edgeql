CREATE MIGRATION m1ruxabzvi5dewmduimtfgb5s4rsahaaxatpzkhm5gdr2nsmcjmpsa
    ONTO m1ir233j3nmz7mgiwbtpl45x54sveujb3c6pxzvej6p3vqua2sftma
{
  ALTER TYPE default::User {
      ALTER LINK banks {
          SET TYPE default::Data USING (.banks[IS default::Data]);
      };
  };
};
