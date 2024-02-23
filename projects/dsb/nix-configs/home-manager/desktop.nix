{ config, pkgs, catppuccin, ... }:
{
  custom-ctps = {
    waybar = builtins.fetchGit{url="https://github.com/catppuccin/waybar.git";rev="https://github.com/catppuccin/waybar/commit/f74ab1eecf2dcaf22569b396eed53b2b2fbe8aff";}
  };

  wayland.windowManager.sway = {
    enable = true;
    package = pkgs.swayfx;
  };

  programs = {
    waybar = {
      enable = true;
      config = {
        modules-left = ["sway/workspaces" "privacy" "tray"];
        modules-center = ["sway/window"];
        modules-right = [ "backlight" "pulseaudio" "cpu" "temperature" "memory" "network" "battery" "clock"];
      };
      style = ''
        @import ${waybar}/themes/mocha.css
      '';
    }
  }
}
