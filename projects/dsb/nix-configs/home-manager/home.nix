{ config, pkgs, catppuccin, ... }:
{
  home.username = "misile";
  home.homeDirectory = "/home/misile";
  home.stateVersion = "23.11"; # dont change it

  home.packages = with pkgs; [
    # Temporary
    gh

    # Development
    git

    # Network
    dhcpcd

    # Fonts
    fira-code-nerdfont nanum
  ] ++ if config.desktop then [swww];

  home.file = {};

  home.sessionVariables = {
    QT_QPA_PLATFORM = "wayland";
    EDITOR = "nvim";
  };

  catppuccin.flavour = "mocha";
  fonts.fontconfig.enable = true;

  # Let Home Manager install and manage itself.
  programs.home-manager.enable = true;
}
