{pkgs, ...}: {
  home.packages = with pkgs; [
    mullvad-vpn pass-wayland tor-browser-bundle-bin monero-gui (python312Packages.mat2)
  ];
}
