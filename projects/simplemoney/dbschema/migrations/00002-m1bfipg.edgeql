CREATE MIGRATION m1bfipg4oixb2kbflckgik24zcvtzyfgkhvr7wnnsiodhmgfzf7sea
    ONTO m17qeo6gyqsa34hlvaixzvblxo4lxw3nmekdahe2xwy4nwr5zvmtoq
{
  ALTER TYPE default::Transaction {
      DROP LINK received;
  };
  ALTER TYPE default::Transaction {
      DROP LINK sent;
  };
  ALTER TYPE default::Transaction {
      CREATE MULTI PROPERTY received: std::str;
  };
  ALTER TYPE default::Transaction {
      CREATE MULTI PROPERTY sent: std::str;
  };
};
