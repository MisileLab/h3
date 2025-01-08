CREATE MIGRATION m1ytl37z7gzmfpfllatt4ml3p5h4wzhqulrs4syhqluwqt5qsbxwcq
    ONTO m1bb5ulhzdcsfyzti4t4ufhxjfu4radkbxedalcgx6f7b2vr63xeoq
{
  ALTER TYPE default::User {
      DROP PROPERTY credit;
  };
  ALTER TYPE default::BaseUser {
      CREATE REQUIRED PROPERTY credit: std::int64 {
          SET default := 0;
          CREATE CONSTRAINT std::min_value(0);
      };
  };
};
