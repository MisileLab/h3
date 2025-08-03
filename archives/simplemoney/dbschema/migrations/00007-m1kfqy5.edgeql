CREATE MIGRATION m1kfqy5gzu6ztvbraihz6sqqgddcmskdm5vaxaczac5dax526fqt4q
    ONTO m1qkrr7rx6sm7j6dvrylvn36xbwazsdx2rxrepuokwjmztvijktwtq
{
  ALTER TYPE default::Product {
      CREATE REQUIRED PROPERTY max_amount: std::int64 {
          SET REQUIRED USING (1000);
          CREATE CONSTRAINT std::min_value(0);
      };
  };
};
