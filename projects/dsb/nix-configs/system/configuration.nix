# Edit this configuration file to define what should be installed on
# your system. Help is available in the configuration.nix(5) man page, on
# https://search.nixos.org/options and in the NixOS manual (`nixos-help`).

{ config, lib, pkgs, ... }:
{
  imports = [] ++ (if pkgs.system == "aarch64-linux" then [
      ./apple-silicon/hardware-configuration.nix
      ./apple-silicon/home.nix
    ] else [
      ./x86_64/home.nix
      ./x86_64/hardware-configuration.nix
    ]);
  # Use the systemd-boot EFI boot loader.
  boot.loader.systemd-boot.enable = true;
  networking.networkmanager.enable = true;

  # Set your time zone.
  time.timeZone = "Asia/Seoul";

  users.users.misile = {isNormalUser = true;extraGroups=["wheel"];};
  system.copySystemConfiguration = true;
  nix.settings.experimental-features = ["nix-command" "flakes"];

  networking.wireless.iwd = {enable = true;settings.General.EnableNetworkConfiguration=true;};
  system.stateVersion = "24.05"; # Just dont touch this
}

