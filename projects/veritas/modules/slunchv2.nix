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

      neis_api_key = lib.mkOption {
        type = lib.types.str;
        default = "";
      };

      admin_key = lib.mkOption {
        type = lib.types.str;
        default = "";
      };
    };

    config = lib.mkIf cfg.enable {
      assertions = [{
        assertion = cfg.admin_key != "" && cfg.neis_api_key != "";
        message = "keys not configured";
      }];
      systemd.services.slunchv2 = lib.mkIf (cfg.enable) {
        description = "Backend of slunchv2";
        wantedBy = [ "multi-user.target" "network-online.target" ];
        script = "exec steam-run ${lib.getExe cfg.package} --port ${cfg.port} ${cfg.envFile}";
        serviceConfig = {
          Restart = "always";
          RestartSec = 5;
          Environment = lib.mkMerge [
            "NEIS_API_KEY=${cfg.neis_api_key}"
            "ADMIN_KEY=${cfg.admin_key}"
            "PORT=${cfg.port}"
          ];
        };
      };
    };
  }
