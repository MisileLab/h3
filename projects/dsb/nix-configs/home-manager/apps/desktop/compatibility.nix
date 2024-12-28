{pkgs, ...}: {
  # https://github.com/NixOS/nixpkgs/issues/367772
  home.packages = with pkgs; [
    wineWowPackages.stable appimage-run /*bottles*/ libreoffice
  ];
}
