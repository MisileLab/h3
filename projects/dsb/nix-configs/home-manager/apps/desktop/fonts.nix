{pkgs, stablep, ...}: {
  fonts.fontconfig.enable = true;
  home.packages = with pkgs; [
    nerd-fonts.fira-code nanum pretendard (stablep.noto-fonts-color-emoji)
    noto-fonts noto-fonts-cjk-sans
  ];
}
