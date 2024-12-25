{pkgs, stablep, ...}: {
  home.packages = with pkgs; [
    wineWowPackages.stable appimage-run (stablep.bottles) libreoffice
  ];
}
