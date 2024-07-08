{pkgs, ...}: {
  home.packages = with pkgs; [
    wineWowPackages.stable appimage-run libreoffice scrcpy
    bottles
  ];
}
