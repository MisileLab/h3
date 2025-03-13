# Edit this configuration file to define what should be installed on
# your system. Help is available in the configuration.nix(5) man page, on
# https://search.nixos.org/options and in the NixOS manual (`nixos-help`).

{ pkgs, ... }:
{
  boot = {
    loader = {
      efi.canTouchEfiVariables = true;
      systemd-boot.enable = true;
    };
    kernelPackages = with pkgs; linuxPackages_latest;
    kernelParams = ["amdgpu.sg_display=0" "initcall_blacklist=amd_pstate_init" "amd_pstate.enable=0"];
    initrd.kernelModules = ["amdgpu"];
    supportedFilesystems = ["ntfs" "btrfs" "ext4" "mtpfs"];
    kernel.sysctl = {
      "net.core.rmem_max" = 4194304;
      "net.core.wmem_max" = 1048576;
    };
  };

  zramSwap = {
    enable = true;
    memoryPercent = 100;
  };

  hardware = {
    graphics = {
      enable = true;
      enable32Bit = true;
      extraPackages = with pkgs; [mesa];
    };
    amdgpu.amdvlk.enable = true;
    enableAllFirmware = true;
    enableRedistributableFirmware = true;
    firmware = with pkgs; [alsa-firmware sof-firmware];
    bluetooth = {enable = true;powerOnBoot = true;};
  };

  networking.networkmanager.enable = true;
  xdg.portal = {
    enable = true;
    extraPortals = with pkgs; [xdg-desktop-portal-gtk xdg-desktop-portal-wlr];
    config.common.default = ["gtk" "wlr"];
  };

  # Set your time zone.
  time.timeZone = "Asia/Seoul";

  users = {
    users.misile = {
      isNormalUser = true;
      extraGroups = ["wheel" "docker"];
      shell = pkgs.nushell;
    };
    extraGroups.vboxusers.members = [ "user-with-access-to-virtualbox" ];
    motd = "I use nixos btw";
  };

  nix = {
    gc = {
      automatic = true;
      options = "--delete-older-than 1d";
    };
    settings = {
      experimental-features = ["nix-command" "flakes"];
      trusted-users = ["root" "misile"];
    };
  };
  programs.appimage.binfmt = true;

  system.stateVersion = "24.11";
}

