{pkgs, ...}: {
  home.packages = with pkgs; [
    yubikey-manager-qt mullvad-vpn pass-wayland tor-browser-bundle-bin
    monero-gui (python312Packages.mat2)
  ];
}
