{pkgs, ...}:
{
  home = {
    packages = with pkgs; [
      hdparm hyperfine hydra-check usbutils
    ];
  };
  programs = {
    btop = {
      enable = true;
      package = pkgs.btop-rocm;
    };
  };
}
