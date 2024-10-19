{pkgs, ...}: {
  fonts.fontconfig.enable = true;
  home.packages = with pkgs; [
    fira-code-nerdfont nanum pretendard noto-fonts-color-emoji
    noto-fonts noto-fonts-cjk-sans
  ];
}
