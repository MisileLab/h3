CREATE MIGRATION m15nfokkoayn73hr756lkgzxyxscwbxhzpzsaoybuwffvf6nnmxsza
    ONTO m1xk7jq7ughuhlssadsvu3gj7cua5wmmssj23p52cgwjioctorhonq
{
  ALTER TYPE default::User {
      DROP PROPERTY name;
  };
};
