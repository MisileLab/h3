{pkgs, ...}: {
  home.packages = with pkgs; [
    mullvad-vpn tor-browser-bundle-bin (python313Packages.mat2)
  ];
}
