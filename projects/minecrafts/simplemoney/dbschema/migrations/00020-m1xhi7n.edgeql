CREATE MIGRATION m1xhi7n6fbbltnx6vqwe6jxdpjxv43ggreex4lp65nyzva3o7xbbuq
    ONTO m1qtzflgegcql6ragctecygv5z4kziwqe3elem6dkszp5gxrof3oha
{
  ALTER TYPE default::Loan {
      CREATE REQUIRED PROPERTY interest: std::int64 {
          SET REQUIRED USING (<std::int64>{1});
      };
  };
  ALTER TYPE default::Product {
      ALTER PROPERTY interest {
          SET TYPE std::int64 USING (<std::int64>.interest);
      };
  };
};
