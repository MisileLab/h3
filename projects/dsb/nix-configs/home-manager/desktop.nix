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
      style = ''
        @import "${custom-ctps.waybar}/themes/mocha.css";
        window#waybar {
          background: rgba(108, 112, 134, 0.21);
          border-radius: 16px;
          box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
          backdrop-filter: blur(5px);
          border: 1px solid rgba(108, 112, 134, 0.3);
        }
      '';
    };
  };
}
