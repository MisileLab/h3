CREATE MIGRATION m1crirjlnfv2c7lblm7o2fve3pko7qlekhtjxfkhz62hefxc3gg2fq
    ONTO m13tjisyuvrhneiyghynl2vg2grf5uixcgpdpwh4zycmego4yhdrea
{
  ALTER TYPE theresa::Letter {
      CREATE REQUIRED PROPERTY file: std::str {
          SET REQUIRED USING (<std::str>{});
      };
  };
};
