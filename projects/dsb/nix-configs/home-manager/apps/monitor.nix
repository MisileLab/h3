{pkgs, ...}: 
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
  programs = {
    btop = {
      enable = true;
      package = pkgs.btop.override {rocmSupport=true;};
      catppuccin.enable = true;
    };
  };
}
