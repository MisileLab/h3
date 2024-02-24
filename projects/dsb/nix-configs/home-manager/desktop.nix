{ config, pkgs, catppuccin, ... }:
let 
  custom-ctps = {
    waybar = builtins.fetchGit{url="https://github.com/catppuccin/waybar.git";rev="f74ab1eecf2dcaf22569b396eed53b2b2fbe8aff";};
  };
  returnColorCSS = {r, g, b, a, addi ? ""}: ''
    /*backdrop-filter: blur(5px);*/
    background: rgba(${toString r}, ${toString g}, ${toString b}, ${toString a});
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    border-radius: 0;
    ${addi}
  '';
in
{
  home.pointerCursor = {
    name = "Adwaita";
    package = pkgs.gnome.adwaita-icon-theme;
    size = 32;
  };

  wayland.windowManager.sway = {
    enable = true;
    config = {
      bars = [];
      startup = [
        {command = "waybar";}
        {command = "${pkgs.swww}/bin/swww-daemon";}
        {command = "bash -c 'sleep 1&&${pkgs.swww}/bin/swww img ~/.config/home-manager/bg.png'";}
      ];
    };
  };

  programs = {
    waybar = {
      enable = true;
      settings = [{
        modules-left = [ "sway/workspaces" "tray"];
        modules-center = [ "sway/window" ];
        modules-right = [ "backlight" "pulseaudio" "cpu" "temperature" "memory" "network" "battery" "clock"];
      }];
      # now we need to make it component
      style = ''
        @import "${custom-ctps.waybar}/themes/mocha.css";
        * {
          font-family: 'Fira Code', monospace;
        }
        window#waybar {${returnColorCSS({r=108;g=112;b=134;a=0.4;})}}
        #workspaces button {
          ${returnColorCSS({r=108;g=112;b=134;a=0.4;addi=''
            padding-left: 6px;
            padding-right: 6px;
          '';})}
        }
        #workspaces button.active {
          ${returnColorCSS({r=147;g=153;b=178;a=0.6;})}
        }
        #workspace button.urgent {
          background: rgba(249, 226, 175, 0.6);
        }
        #workspace button.hover {
          ${returnColorCSS({r=127;g=132;b=156;a=0.6;addi=''
            padding-left: 6px;
            padding-right: 6px;
          '';})}
        }
        .modules-left * {
          color: @subtext1;
        }
        .modules-center * {
          color: @text;
        }
        .modules-right * {
          ${returnColorCSS({r=147;g=153;b=178;a=0.6;addi=''
            padding-left: 6px;
            padding-right: 6px;
            color: @subtext1;
          '';})};
        }
        #battery.warning {
          background: rgba(249, 226, 175, 0.6);
        }
        #battery.critical {
          background: rgba(243, 139, 168, 0.6);
        }
      '';
    };
  };
}
