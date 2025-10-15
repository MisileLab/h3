{ ... }:
{
  imports = [
    ./apps/base.nix
    ./apps/desktop/base.nix
  ];
  home = {
    username = "misile";
    homeDirectory = "/home/misile";
    stateVersion = "25.05";
    sessionVariables = {
      QT_QPA_PLATFORM = "wayland";
      EDITOR = "nvim";
      XDG_SCREENSHOTS_DIR = "/home/misile/screenshots";
    };
  };

  catppuccin = {
    flavor = "mocha";
    enable = true;
  };
  nixpkgs.config.allowUnfree = true;
  gtk.enable = true;

  # Let Home Manager install and manage itself.
  programs.home-manager.enable = true;
}
