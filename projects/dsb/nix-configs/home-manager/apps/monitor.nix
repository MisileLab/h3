{pkgs, ...}: 
let
  bvtop = pkgs.makeDesktopItem {
    name = "bvtop";
    desktopName = "bvtop";
    icon = "btop";
    exec = "${pkgs.alacritty}/bin/alacritty -e zellij --layout /home/misile/non-nixos-things/bvtop.kdl";
  };
in
{
  home = {
    packages = with pkgs; [
      bvtop hdparm hyperfine nvtopPackages.amd hydra-check usbutils
    ];
    file = {
      "non-nixos-things/bvtop.kdl".text = "
        layout {
          tab {
            pane command=\"btop\"
          }
          tab {
            pane command=\"nvtop\"
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
      catppuccin.enable = true;
    };
  };
}