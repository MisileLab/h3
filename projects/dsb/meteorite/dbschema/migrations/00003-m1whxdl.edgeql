CREATE MIGRATION m1whxdlmn6xj76pcdsk7v4lsjxu27ypffdscl26dstwkk33tcgln2a
    ONTO m1f4fybdqj6riqmuwekoiervwdpgaouo7ouxo4vo47tusspew44u7a
{
  ALTER TYPE Lunch::School {
      ALTER PROPERTY ofcdc_code {
          DROP CONSTRAINT std::exclusive;
      };
  };
};
