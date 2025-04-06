{config, pkgs, ...}:
let
  writeScript = name: content: pkgs.writeShellScriptBin name "${content} $@";
in
  {
    imports = [
      ./development.nix
      ./utils.nix
    ];
    home = {
      packages = with pkgs; [
        smartmontools borgbackup miniserve openssl
        ansifilter b3sum age sops
        (writeScript "nix-clean" "nix-collect-garbage -d && sudo nix-collect-garbage -d && sudo nix store optimise")
      ];
    };
    sops = {
      age.keyFile = "/home/veritas/.config/sops/age/keys.txt";
      defaultSopsFile = ../secrets.yaml;
      defaultSecretsMountPoint = "/run/user/1000/secrets.d";
      secrets = {
        NEIS_API_KEY.path = "${config.sops.defaultSymlinkPath}/neis_api_key";
        ADMIN_KEY.path = "${config.sops.defaultSymlinkPath}/admin_key";
      };
    };
    programs = {
      bash.enable = true;
      topgrade.enable = true;
    };
    services.pueue.enable = true;
  }
