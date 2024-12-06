{pkgs, stablep, ...}: {
  home.packages = with pkgs; [
    dhcpcd cloudflare-warp (stablep.trayscale)
    nethogs (pkgs.writeShellScriptBin "nhs" "sudo ${pkgs.nethogs}/bin/nethogs -b wg0-mullvad")
  ];
  programs = {
    irssi.enable = true;
  };
}
