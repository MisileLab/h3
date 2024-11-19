{pkgs, stablep, ...}: {
  fonts.fontconfig.enable = true;
  home.packages = with pkgs; [
    fira-code-nerdfont nanum pretendard noto-fonts-color-emoji
    (stablep.noto-fonts) noto-fonts-cjk-sans
  ];
}
