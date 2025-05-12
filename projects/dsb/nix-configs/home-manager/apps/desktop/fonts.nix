{pkgs, ...}: {
  fonts.fontconfig.enable = true;
  home.packages = with pkgs; [
    nerd-fonts.fira-code nanum pretendard noto-fonts-color-emoji
    noto-fonts noto-fonts-cjk-sans
  ];
}
