{ config, lib, pkgs, ... }:
{
  imports =
    [
      ./common.nix
      ./hardware-configuration.nix
    ];
  # i need to do lanzaboote but im lazy
  boot.loader.efi.canTouchEfiVariables = true;
  boot.kernelPackages = pkgs.linuxPackages_latest;
  boot.initrd.kernelModules = ["amdgpu"];
  boot.supportedFilesystems = [ "ntfs" "btrfs" "ext4" ];

  hardware = {
    opengl = {enable = true; driSupport = true;};
    pulseaudio.enable = true;
    enableAllFirmware = true;
    enableRedistributableFirmware = true;
  };

  time.hardwareClockInLocalTime = true;

  nix.gc = {
    automatic = true;
    options = "--delete-older-than 1d";
  };
  services = {
    printing.enable = true;
    avahi = {
      enable = true;
      nssmdns = true;
      openFirewall = true;
    };
    clamav = {
      daemon.enable = true;
      updater.enable = true;
    };
  };

  virtualisation = {
    docker.enable = true;
    libvirtd = {
      enable = true;
      qemu = {
        package = pkgs.qemu_kvm;
        runAsRoot = true;
        swtpm.enable = true;
        ovmf = {
          enable = true;
          packages = [(pkgs.OVMF.override {
            secureBoot = true;
            tpmSupport = true;
          }).fd];
        };
      };
    };
    waydroid.enable = true;
  };

  nixpkgs.config.allowUnfree = true;
  environment.systemPackages = with pkgs; [kdiskmark];
  programs.steam.enable = true;
}
