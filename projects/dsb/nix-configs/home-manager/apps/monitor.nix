{pkgs, stablep, ...}: 
let
  bvtop = pkgs.makeDesktopItem {
    name = "abtop";
    desktopName = "abtop";
    icon = "btop";
    exec = "${pkgs.alacritty}/bin/alacritty -e zellij --layout /home/misile/non-nixos-things/abtop.kdl";
  };
in
{
  home = {
    packages = with pkgs; [
      bvtop hdparm hyperfine hydra-check usbutils
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
  programs = {
    btop = {
      enable = true;
      package = stablep.btop.override {rocmSupport=true;};
      catppuccin.enable = true;
    };
  };
}
