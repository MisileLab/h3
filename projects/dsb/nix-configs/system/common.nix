# Edit this configuration file to define what should be installed on
# your system. Help is available in the configuration.nix(5) man page, on
# https://search.nixos.org/options and in the NixOS manual (`nixos-help`).

{ config, lib, pkgs, ... }:
{
  boot = {
    loader = {
      efi.canTouchEfiVariables = true;
      systemd-boot.enable = true;
    };
    kernelPackages = with pkgs; linuxPackages_latest;
    initrd.kernelModules = ["amdgpu"];
    supportedFilesystems = ["ntfs" "btrfs" "ext4"];
  };

  hardware = {
    opengl = {enable = true; driSupport = true;};
    enableAllFirmware = true;
    enableRedistributableFirmware = true;
    firmware = with pkgs; [alsa-firmware sof-firmware];
    bluetooth = {enable = true;powerOnBoot = true;};
  };

  networking = {
    networkmanager.enable = true;
    wireless.iwd = {
      enable = true;
      settings.General.EnableNetworkConfiguration=true;
    };
  };

  xdg.portal = {
    enable = true;
    extraPortals = with pkgs; [xdg-desktop-portal-gtk xdg-desktop-portal-wlr];
    config.common.default = ["gtk" "wlr"];
  };

  # Set your time zone.
  time = {
    timeZone = "Asia/Seoul";
    hardwareClockInLocalTime = true;
  };

  users.users.misile = {
    isNormalUser = true;
    extraGroups=["wheel" "docker"];
  };

  nix.settings.experimental-features = ["nix-command" "flakes"];
  sound.enable = false;

  system = {
    copySystemConfiguration = true;
    stateVersion = "24.05";
  };
}

