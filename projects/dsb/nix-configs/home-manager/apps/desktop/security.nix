{pkgs, ...}: {
  # https://github.com/NixOS/nixpkgs/issues/352598
  home.packages = with pkgs; [
    /* yubikey-manager-qt */ mullvad-vpn pass-wayland tor-browser-bundle-bin
    monero-gui (python312Packages.mat2)
  ];
}
