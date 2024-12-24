{pkgs, stablep, ...}: {
  home.packages = with pkgs; [
    wineWowPackages.stable appimage-run scrcpy (stablep.bottles) libreoffice
  ];
}
