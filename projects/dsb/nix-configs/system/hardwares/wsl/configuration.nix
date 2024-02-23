{ config, lib, pkgs, ... }:
{
  imports = [
    <nixos-wsl/modules>
    ../../common.nix
  ];

  wsl.enable = true;
  wsl.defaultUser = "nixos";

  system.stateVersion = "23.11"; # nou
}