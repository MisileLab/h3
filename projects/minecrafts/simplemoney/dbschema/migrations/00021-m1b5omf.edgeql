CREATE MIGRATION m1b5omfqr7w3qdns3awgyyfdfhtddj44bvqadepvobmmjnuqbgahrq
    ONTO m1xhi7n6fbbltnx6vqwe6jxdpjxv43ggreex4lp65nyzva3o7xbbuq
{
  ALTER TYPE default::Loan {
      ALTER PROPERTY interest {
          CREATE CONSTRAINT std::min_value(0);
      };
  };
};
