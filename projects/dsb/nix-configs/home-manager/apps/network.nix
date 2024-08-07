{pkgs, ...}: {
  home.packages = with pkgs; [
    dhcpcd cloudflare-warp
    nethogs (pkgs.writeShellScriptBin "nhs" "sudo ${pkgs.nethogs}/bin/nethogs -b wg0-mullvad")
    trayscale
  ];
  programs = {
    irssi.enable = true;
  };
}
