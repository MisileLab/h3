{pkgs, stablep, ...}: {
  fonts.fontconfig.enable = true;
  home.packages = with pkgs; [
    nerd-fonts.fira-code nanum pretendard noto-fonts-color-emoji
    (stablep.noto-fonts) noto-fonts-cjk-sans
  ];
}
