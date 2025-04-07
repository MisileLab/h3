{config, lib, pkgs, ...}:
let
  cfg = config.services.slunchv2;
in 
  {
    options.services.slunchv2 = {
      enable = lib.mkEnableOption "slunchv2";
      package = lib.mkPackageOption pkgs "slunchv2" {};

      port = lib.mkOption {
        type = lib.types.ints.unsigned;
        default = 80;
      };

      neisApiKeyPath = lib.mkOption {
        type = lib.types.path;
        default = null;
      };

      adminKeyPath = lib.mkOption {
        type = lib.types.path;
        default = null;
      };
    };

    config = lib.mkIf cfg.enable {
      systemd.services.slunchv2 = lib.mkIf (cfg.enable) {
        description = "Backend of slunchv2";
        wantedBy = [ "multi-user.target" "network-online.target" ];
        script = "
          #!/bin/sh
          export NEIS_API_KEY=$(cat ${cfg.neisApiKeyPath})
          export ADMIN_KEY=$(cat ${cfg.adminKeyPath})
          export PORT=${toString cfg.port}
          exec ${pkgs.steam-run}/bin/steam-run ${lib.getExe cfg.package}
        ";
        serviceConfig = {
          Restart = "always";
          RestartSec = 5;
        };
      };
    };
  }
