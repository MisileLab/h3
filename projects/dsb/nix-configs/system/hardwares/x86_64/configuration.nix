{ config, lib, pkgs, ... }:
{
  imports =
    [
      ../../common.nix
      ./hardware-configuration.nix
    ];
  # i need to do lanzaboote but im lazy
  boot.loader.efi.canTouchEfiVariables = true;
  boot.kernelPackages = pkgs.linuxPackages_latest;
  boot.initrd.kernelModules = ["amdgpu"];
  boot.supportedFilesystems = [ "ntfs" "btrfs" "ext4" ];
  hardware.opengl = {enable = true; driSupport = true;};
  nix.gc.automatic = true;
  nix.gc.options = "--delete-older-than 1d";
  time.hardwareClockInLocalTime = true;
  hardware.pulseaudio.enable = true;
  services.printing.enable = true;
  services.avahi = {
    enable = true;
    nssmdns = true;
    openFirewall = true;
  };
  virtualisation.docker.enable = true;
  virtualisation.libvirtd = {
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
  services.clamav = {
    daemon.enable = true;
    updater.enable = true;
  };
  nixpkgs.config.allowUnfree = true;
  environment.systemPackages = with pkgs; [kdiskmark];
  
  # desktop-specific
  programs.steam.enable = true;
  virtualisation.waydroid.enable = true;
}
