{pkgs, stablep, ...}: {
  # https://github.com/NixOS/nixpkgs/issues/436421
  home.packages = with pkgs; [
    tor-browser-bundle-bin (stablep.python313Packages.mat2)
  ];
}
