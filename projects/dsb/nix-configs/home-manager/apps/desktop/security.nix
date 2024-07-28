{pkgs, stablep, ...}: {
  home.packages = with pkgs; [
    yubikey-manager-qt mullvad-vpn pass-wayland tor-browser-bundle-bin
    metadata-cleaner
  ] ++ (with stablep; [monero-gui]);
}
