{pkgs, ...}:
{
  home = {
    packages = with pkgs; [
      hdparm hyperfine hydra-check usbutils
    ];
  };
  programs = {
    # https://github.com/NixOS/nixpkgs/pull/367695
    btop.enable = true;
  };
}
