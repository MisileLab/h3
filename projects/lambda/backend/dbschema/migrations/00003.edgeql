CREATE MIGRATION m12y4yy5lppcz5xavjqg2uyxlzryfmuos6b5h6u2g5gmdwhsg4n4ra
    ONTO m1bqumzxw3x3hwn73cnn6h55jwvrmuuli75udavcrn2syrhvnxeulq
{
  ALTER TYPE default::User {
      CREATE REQUIRED PROPERTY time: std::float64 {
          SET REQUIRED USING (<std::float64>{});
      };
  };
};
