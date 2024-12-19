{pkgs, ...}: {
  home.packages = with pkgs; [
    mullvad-vpn pass-wayland tor-browser-bundle-bin (python312Packages.mat2)
  ];
}
