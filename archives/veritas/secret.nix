{
  sops = {
    age.keyFile = "/etc/sops/key.txt";
    defaultSopsFile = ./secrets.yaml;
    secrets = {
      neisApiKey = {
        restartUnits = [ "slunchv2.service" ];
      };
      adminKey = {};
    };
  };
}
