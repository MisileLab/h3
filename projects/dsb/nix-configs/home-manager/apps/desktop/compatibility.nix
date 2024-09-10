{pkgs, stablep, ...}: {
  home.packages = with pkgs; [
    (stablep.wineWowPackages.stable) appimage-run scrcpy bottles libreoffice
  ];
}
