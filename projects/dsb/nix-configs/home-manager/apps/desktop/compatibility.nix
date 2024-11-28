{pkgs, stablep, ...}: {
  home.packages = with pkgs; [
    (wineWowPackages.stable) appimage-run scrcpy bottles (stablep.libreoffice)
  ];
}
