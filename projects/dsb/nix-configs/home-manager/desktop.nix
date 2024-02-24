{ config, pkgs, catppuccin, ... }:
  let custom-ctps = {
    waybar = builtins.fetchGit{url="https://github.com/catppuccin/waybar.git";rev="f74ab1eecf2dcaf22569b396eed53b2b2fbe8aff";};
  };
in
{
  wayland.windowManager.sway = {
    enable = true;
    config = {
      bars = [];
      startup = [
        {command = "waybar";}
        {command = "${pkgs.swww}/bin/swww-daemon";}
        # {command = "bash -c 'sleep 2&&${pkgs.swww}/bin/swww img ~/bg.jpg'";}
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
      # cursor size multiply 4
      # change text color
      style = ''
        @import "${custom-ctps.waybar}/themes/mocha.css";
        window#waybar {
          /*backdrop-filter: blur(5px);*/
          background: rgba(108, 112, 134, 0.4);
          box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
          border: 1px solid rgba(108, 112, 134, 0.3);
        }
        * {
          font-family: 'Fira Code', monospace;
        }
        #workspaces button {
          /*backdrop-filter: blur(5px);*/
          background: rgba(127, 132, 156, 0.6);
          border-radius: 0;
          box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
          border: 1px solid rgba(127, 132, 156, 0.3);
          padding-left: 6px;
          padding-right: 6px;
        }
        #workspaces button.active {
          background: rgba(147, 153, 178, 0.6);
          border-radius: 0px;
          box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
          /*backdrop-filter: blur(5px);*/
          border: 1px solid rgba(147, 153, 178, 0.3);
        }
        #workspace button.urgent {
          background: rgba(249, 226, 175, 0.6);
        }
        .modules-right * {
          background: rgba(147, 153, 178, 0.6);
          border-radius: 0;
          box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
          /* backdrop-filter: blur(5px); */
          border: 1px solid rgba(147, 153, 178, 0.3);
          padding-left: 6px;
          padding-right: 6px;
        }
      '';
    };
  };
}
