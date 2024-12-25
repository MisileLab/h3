{pkgs, stablep, ...}:
{
  home = {
    packages = with pkgs; [
      hdparm hyperfine hydra-check usbutils
    ];
  };
  catppuccin.btop.enable = true;
  programs = {
    btop = {
      enable = true;
      package = stablep.btop.override {rocmSupport=true;};
    };
  };
}
