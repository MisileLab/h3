# Edit this configuration file to define what should be installed on
# your system. Help is available in the configuration.nix(5) man page, on
# https://search.nixos.org/options and in the NixOS manual (`nixos-help`).

{ config, lib, pkgs, ... }:
{
  imports =
    [
      ./hardware-configuration.nix
      <apple-silicon-support/apple-silicon-support>
      ./hardware.nix
    ];
  # Use the systemd-boot EFI boot loader.
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = false;

  networking.networkmanager.enable = true;

  # Set your time zone.
  time.timeZone = "Asia/Seoul";

  users.users.misile = {isNormalUser = true;extraGroups=["wheel"];};
  system.copySystemConfiguration = true;
  nix.settings.experimental-features = ["nix-command" "flakes"];

  networking.wireless.iwd = {enable = true;settings.General.EnableNetworkConfiguration=true;};
  programs.light.enable = true;
  hardware.asahi.useExperimentalGPUDriver = true;
  services.actkbd={enable=true;bindings=[{keys=[225];events=["key"];command="/run/current-system/sw/bin/light -A 10";} {keys=[224];events=["key"];command="/run/current-system/sw/bin/light -U 10";}];};
  system.stateVersion = "24.05"; # Just dont touch this

}

