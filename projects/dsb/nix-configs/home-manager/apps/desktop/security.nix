{pkgs, stablep, ...}: {
  home.packages = with pkgs; [
    yubikey-manager-qt mullvad-vpn pass-wayland tor-browser-bundle-bin
    monero-gui
    # https://github.com/NixOS/nixpkgs/issues/348081
    (stablep.metadata-cleaner)
  ];
}
