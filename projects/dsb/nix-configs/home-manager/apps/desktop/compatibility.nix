{pkgs, stablep, ...}: {
  home.packages = with pkgs; [
    (stablep.wineWowPackages.stable) appimage-run (stablep.bottles) libreoffice
  ];
}
