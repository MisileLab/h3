{...}:
{
  programs = {
    neovim = {
      enable = true;
      vimAlias = true;
    };
    lazygit.enable = true;
    git = {
      enable = true;
      lfs.enable = true;
      extraConfig = {
        safe.directory = "*";
        core.editor = "nvim";
      };
    };
    direnv = {
      enable = true;
      nix-direnv.enable = true;
    };
  };
}
