{ config, pkgs, catppuccin, ... }:
  let custom-ctps = {
    waybar = builtins.fetchGit{url="https://github.com/catppuccin/waybar.git";rev="f74ab1eecf2dcaf22569b396eed53b2b2fbe8aff";};
  };
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
    package = pkgs.swayfx;
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
        window#waybar {
          /*backdrop-filter: blur(5px);*/
          background: rgba(108, 112, 134, 0.4);
          box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }
        * {
          font-family: 'Fira Code', monospace;
        }
        #workspaces button {
          /*backdrop-filter: blur(5px);*/
          background: rgba(127, 132, 156, 0.6);
          border-radius: 0;
          box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
          padding-left: 6px;
          padding-right: 6px;
        }
        #workspaces button.active {
          background: rgba(147, 153, 178, 0.6);
          border-radius: 0px;
          box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
          /*backdrop-filter: blur(5px);*/
        }
        #workspace button.urgent {
          background: rgba(249, 226, 175, 0.6);
        }
        #workspace button.hover {
          /*backdrop-filter: blur(5px);*/
          background: rgba(127, 132, 156, 0.6);
          border-radius: 0;
          box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
          padding-left: 6px;
          padding-right: 6px;
        }
        .modules-left * {
          text-color: @subtext1;
        }
        .modules-center * {
          text-color: @text;
        }
        .modules-right * {
          background: rgba(147, 153, 178, 0.6);
          border-radius: 0;
          box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
          /* backdrop-filter: blur(5px); */
          padding-left: 6px;
          padding-right: 6px;
          text-color: @subtext1;
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
