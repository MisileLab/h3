CREATE MIGRATION m1yrg4gjuugc6vqekwjch6tr5nywjv4kgyx46dpbuynitxctceflxa
    ONTO m1fhzhlws46et2kudwb6fcmmyxasske2orr5f37na3bdpc6mwcdinq
{
  ALTER TYPE default::Bank {
      ALTER PROPERTY name {
          SET REQUIRED USING (<std::str>{});
      };
  };
};
