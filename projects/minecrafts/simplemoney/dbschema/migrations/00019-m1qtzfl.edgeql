CREATE MIGRATION m1qtzflgegcql6ragctecygv5z4kziwqe3elem6dkszp5gxrof3oha
    ONTO m1s2742bkzxsjqvnwzojok2533blsey3fb3ukgr5iu6irobhnyfcaa
{
  ALTER TYPE default::Product {
      DROP LINK bank;
  };
};
