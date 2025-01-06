CREATE MIGRATION m1jpkgehk623rsdzmdmukb6vekmmdqmudpb7uakjapogaia4765pgq
    ONTO m1gt7t4nj7m5wkwgdr57yt25ykrqqpwb4j5l6kzcpm243vyf5ohzvq
{
  ALTER TYPE default::Borrow {
      DROP CONSTRAINT std::exclusive ON ((.receiver, .sender, .name));
  };
  ALTER TYPE default::Borrow {
      CREATE REQUIRED LINK product: default::Product {
          SET REQUIRED USING (<default::Product>{});
      };
  };
  ALTER TYPE default::Borrow {
      CREATE REQUIRED PROPERTY date: std::int64 {
          SET REQUIRED USING (<std::int64>{});
          CREATE CONSTRAINT std::min_value(0);
      };
  };
  ALTER TYPE default::Borrow {
      DROP PROPERTY product_value;
      DROP EXTENDING default::Product;
  };
};
