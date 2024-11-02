{pkgs, stablep, ...}: {
  fonts.fontconfig.enable = true;
  home.packages = with pkgs; [
    fira-code-nerdfont nanum pretendard (stablep.noto-fonts-color-emoji)
    noto-fonts noto-fonts-cjk-sans
  ];
}
