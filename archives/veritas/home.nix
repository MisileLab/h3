{ ... }:
{
  imports = [
    ./apps/base.nix
  ];
  home = {
    username = "veritas";
    homeDirectory = "/home/veritas";
    stateVersion = "24.11";
    sessionVariables.EDITOR = "nvim";
  };

  nixpkgs.config.allowUnfree = true;
  xdg.enable = true;

  # Let Home Manager install and manage itself.
  programs.home-manager.enable = true;
}
