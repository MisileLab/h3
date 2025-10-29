{pkgs, ...}: {
  # https://github.com/NixOS/nixpkgs/pull/436581
  home.packages = with pkgs; [
    tor-browser mat2
  ];
}
