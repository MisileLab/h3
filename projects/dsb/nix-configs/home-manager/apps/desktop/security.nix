{pkgs, ...}: {
  home.packages = with pkgs; [
    tor-browser-bundle-bin (python313Packages.mat2)
  ];
}
