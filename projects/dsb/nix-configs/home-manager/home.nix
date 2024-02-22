{ config, pkgs, ... }:
{
  home.username = "misile";
  home.homeDirectory = "/home/misile";
  home.stateVersion = "23.11"; # dont change it

  home.packages = with pkgs; [
    dhcpcd
  ];

  home.file = {};

  home.sessionVariables = {};

  programs = {
    atuin = {
      enable = true;
      settings = {
        style = "full";
        inline_height = 20;
        auto_sync = true;
        sync_frequency = "1h";
      };
    };
  };

  # Let Home Manager install and manage itself.
  programs.home-manager.enable = true;
}
