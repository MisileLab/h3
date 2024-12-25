{pkgs, stablep, ...}:
let
  abtop = pkgs.makeDesktopItem {
    name = "abtop";
    desktopName = "abtop";
    icon = "btop";
    exec = "${pkgs.kitty}/bin/kitty -e zellij --layout /home/misile/non-nixos-things/abtop.kdl";
  };
in
{
  home = {
    packages = with pkgs; [
      abtop hdparm hyperfine hydra-check usbutils
    ];
    file = {
      "non-nixos-things/abtop.kdl".text = "
        layout {
          tab {
            pane command=\"btop\"
          }
          tab {
            pane command=\"auto-cpufreq\" {
              args \"--stats\"
            }
          }
        }
      ";
    };
  };
  catppuccin.btop.enable = true;
  programs = {
    btop = {
      enable = true;
      package = stablep.btop.override {rocmSupport=true;};
    };
  };
}
