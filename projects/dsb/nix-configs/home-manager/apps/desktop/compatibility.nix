{pkgs, stablep, ...}: {
  home.packages = with pkgs; [
    wineWowPackages.stable appimage-run bottles (stablep.libreoffice)
  ];
}
