{pkgs, ...}: {
  # https://github.com/NixOS/nixpkgs/pull/436581
  home.packages = with pkgs; [
    tor-browser-bundle-bin mat2
  ];
}
