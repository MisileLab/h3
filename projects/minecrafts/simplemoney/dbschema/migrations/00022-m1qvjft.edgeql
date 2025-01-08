CREATE MIGRATION m1qvjft2w2ov6yzx7o6pe7yzvbvd5lmcwbfkrj4bmk6c2sbrtq2zcq
    ONTO m1b5omfqr7w3qdns3awgyyfdfhtddj44bvqadepvobmmjnuqbgahrq
{
  ALTER TYPE default::Loan {
      ALTER PROPERTY date {
          DROP CONSTRAINT std::min_value(0);
          SET TYPE std::datetime USING (std::datetime_current());
      };
  };
};
