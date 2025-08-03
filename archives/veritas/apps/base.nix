{pkgs, ...}:
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
    programs = {
      bash.enable = true;
      topgrade.enable = true;
    };
    services.pueue.enable = true;
  }
