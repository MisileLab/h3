{pkgs, ...}:
{
  home = {
    packages = with pkgs; [
      hdparm hyperfine hydra-check usbutils
    ];
  };
  catppuccin.bottom.enable = true;
  programs = {
    bottom.enable = true;
  };
}
