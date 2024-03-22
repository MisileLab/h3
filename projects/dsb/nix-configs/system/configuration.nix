{ config, lib, pkgs, ... }:
{
  imports = [
    ./common.nix
    ./hardware-configuration.nix
    ./lanzaboote.nix
  ];
  boot = {
    loader.efi.canTouchEfiVariables = true;
    kernelPackages = with pkgs; linuxPackagesFor linux_latest;
    initrd.kernelModules = ["amdgpu"];
    supportedFilesystems = [ "ntfs" "btrfs" "ext4" ];
    extraModprobeConfig = ''
      blacklist snd-soc-dmic
      blacklist snd-acp3x-rn
      blacklist snd-acp3x-pdm-dma
    '';
  };

  hardware = {
    opengl = {enable = true; driSupport = true;};
    enableAllFirmware = true;
    enableRedistributableFirmware = true;
    firmware = with pkgs; [alsa-firmware sof-firmware];
  };

  xdg.portal = {
    enable = true;
    extraPortals = with pkgs; [xdg-desktop-portal-gtk xdg-desktop-portal-wlr];
    config.common.default = ["gtk" "wlr"];
  };
  
  security = {
    rtkit.enable = true;
    pam.services = {
      swaylock.text = ''
        auth sufficient pam_unix.so try_first_pass likeauth nullok
        auth sufficient pam_fprintd.so
        auth include login
      '';
    };
  };

  time.hardwareClockInLocalTime = true;

  nix.gc = {
    automatic = true;
    options = "--delete-older-than 1d";
  };
  services = {
    fwupd.enable = true;
    flatpak.enable = true;
    fprintd.enable = true;
    openvpn.servers = {
      VPN = { config = '' config /home/misile/non-nixos-things/openvpns/profile.ovpn ''; autoStart = false; };
    };
    dbus.enable = true;
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
    auto-cpufreq.enable = true;
    pipewire = {
      enable = true;
      alsa = {enable = true;support32Bit = true;};
      pulse.enable = true;
    };
    tor = {
      enable = true;
      client.enable = true;
    };
    tailscale.enable = true;
    upower.enable = true;
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
    virtualbox.host.enable = true;
    waydroid.enable = true;
  };
  
  users.extraGroups.vboxusers.members = [ "user-with-access-to-virtualbox" ];
  
  systemd = {
    user.services.polkit-gnome-authentication-agent-1 = {
    description = "polkit-gnome-authentication-agent-1";
    wantedBy = [ "graphical-session.target" ];
    wants = [ "graphical-session.target" ];
    after = [ "graphical-session.target" ];
    serviceConfig = {
        Type = "simple";
        ExecStart = "${pkgs.polkit_gnome}/libexec/polkit-gnome-authentication-agent-1";
        Restart = "on-failure";
        RestartSec = 1;
        TimeoutStopSec = 10;
      };
    };
  };

  nixpkgs.config.allowUnfree = true;
  environment.systemPackages = with pkgs; [kdiskmark fprintd auto-cpufreq];
  programs = {
    nix-ld.enable = true;
    steam.enable = true;
    dconf.enable = true;
  };
}
