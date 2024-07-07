{pkgs, stablep, ...}: {
  home.packages = with pkgs; [
    wineWowPackages.stable appimage-run libreoffice scrcpy
  ] ++ (with stablep; [bottles]);
}
