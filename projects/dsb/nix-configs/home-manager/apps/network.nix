{pkgs, ...}: {
  home.packages = with pkgs; [
    dhcpcd cloudflare-warp trayscale
  ];
  programs = {
    irssi.enable = true;
  };
}
