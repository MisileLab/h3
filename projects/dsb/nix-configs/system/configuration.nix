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
    enableAllFirmware = true;
    enableRedistributableFirmware = true;
  };
  
  security.rtkit.enable = true;

  time.hardwareClockInLocalTime = true;

  nix.gc = {
    automatic = true;
    options = "--delete-older-than 1d";
  };
  services = {
    printing.enable = true;
    avahi = {
      enable = true;
      nssmdns4 = true;
      openFirewall = true;
    };
    clamav = {
      daemon.enable = true;
      updater.enable = true;
    };
    tlp.enable = true;
    pipewire = {
      enable = true;
      alsa = {enable = true;support32Bit = true;};
      pulse.enable = true;
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
  programs = {
    nix-ld.enable = true;
    steam.enable = true;
  };
}
