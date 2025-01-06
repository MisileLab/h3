CREATE MIGRATION m1xk7jq7ughuhlssadsvu3gj7cua5wmmssj23p52cgwjioctorhonq
    ONTO m1jpkgehk623rsdzmdmukb6vekmmdqmudpb7uakjapogaia4765pgq
{
  ALTER TYPE default::User {
      ALTER PROPERTY name {
          DROP CONSTRAINT std::exclusive;
      };
  };
};
