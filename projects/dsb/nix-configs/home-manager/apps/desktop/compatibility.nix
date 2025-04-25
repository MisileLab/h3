{pkgs, ...}: {
  home.packages = with pkgs; [
    wineWowPackages.stable appimage-run bottles libreoffice
  ];
}
