CREATE MIGRATION m1yqwn6fnckgwn6yvrudc3vmzxj5dtmryku2g5xrwnkuey4k7imw7q
    ONTO m1crirjlnfv2c7lblm7o2fve3pko7qlekhtjxfkhz62hefxc3gg2fq
{
  ALTER TYPE theresa::User {
      ALTER PROPERTY signature {
          SET TYPE std::str USING ('');
      };
  };
};
