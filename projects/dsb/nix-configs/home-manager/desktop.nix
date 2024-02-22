{ config, pkgs, ... }:
{
  programs = {
    atuin = {
      enable = true;
      enableBashIntegration = true;
    };
  };
}
