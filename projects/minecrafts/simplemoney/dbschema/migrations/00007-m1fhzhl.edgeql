CREATE MIGRATION m1fhzhlws46et2kudwb6fcmmyxasske2orr5f37na3bdpc6mwcdinq
    ONTO m1lzxbk473aiunen4u3rgnpd7a3iwpgtrg6evyehug6bpj2ifsjwfq
{
  ALTER TYPE default::Bank {
      CREATE PROPERTY name: std::str;
  };
};
