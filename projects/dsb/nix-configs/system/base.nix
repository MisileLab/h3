# Edit this configuration file to define what should be installed on
# your system. Help is available in the configuration.nix(5) man page, on
# https://search.nixos.org/options and in the NixOS manual (`nixos-help`).

{ pkgs, ... }:
{
  # https://github.com/NixOS/nix/issues/12420
  nix.package = pkgs.lix;

  boot = {
    loader = {
      efi.canTouchEfiVariables = true;
      systemd-boot.enable = true;
    };
    kernelPackages = pkgs.linuxPackages_latest;
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
    groups.plugdev = {};
    users.misile = {
      isNormalUser = true;
      extraGroups = ["wheel" "docker" "adbusers" "plugdev"];
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
      substituters = ["https://cache.misile.xyz/cache"];
      trusted-public-keys = ["cache:zNeEp8cQiXd3s4UrwBgZQq5/9SnX4W/n06/lsxMzPug="];
      experimental-features = ["nix-command" "flakes" "repl-flake"];
      trusted-users = ["misile"];
    };
  };
  programs = {
    command-not-found.enable = false;
    appimage.binfmt = true;
    adb.enable = true;
  };
  services.udev.packages = [ pkgs.android-udev-rules ];

  system.stateVersion = "24.11";
}

