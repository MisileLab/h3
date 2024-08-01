{ config, pkgs, catppuccin, c, ... }:
let
  nurpkgs = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/master.tar.gz") { inherit pkgs; };
in
{
  imports = [
    ./apps/base.nix
    ./apps/desktop/base.nix
  ];
  home = {
    username = "misile";
    homeDirectory = "/home/misile";
    stateVersion = "24.05";
    sessionVariables = {
      QT_QPA_PLATFORM = "wayland";
      EDITOR = "nvim";
      XDG_SCREENSHOTS_DIR = "/home/misile/screenshots";
    };
  };

  catppuccin.flavor = "mocha";
  nixpkgs.config.allowUnfree = true;
  gtk.enable = true;

  # Let Home Manager install and manage itself.
  programs.home-manager.enable = true;
}
