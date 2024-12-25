{pkgs, ...}: {
  home.packages = with pkgs; [
    dhcpcd
  ];
}
