{ pkgs, ... }:
{
  imports = [
    ./common.nix
    ./hardware-configuration.nix
    ./security.nix
  ];

  nix.gc = {
    automatic = true;
    options = "--delete-older-than 1d";
  };
  services = {
    gnome.gnome-keyring.enable = true;
    pcscd.enable = true;
    chrony = {
      enable = true;
      enableNTS = true;
      servers = ["time.cloudflare.com"];
    };
    pipewire = {
      enable = true;
      alsa = {enable = true;support32Bit = true;};
      pulse.enable = true;
    };
    fwupd.enable = true;
    flatpak.enable = true;
    dbus.enable = true;
    printing.enable = true;
    avahi = {
      enable = true;
      nssmdns4 = true;
      openFirewall = true;
    };
    auto-cpufreq.enable = true;
    upower.enable = true;
    tumbler.enable = true;
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

  nixpkgs.config.allowUnfree = true;
  environment.systemPackages = with pkgs; [kdiskmark auto-cpufreq];
  programs = {
    nix-ld.enable = true;
    steam.enable = true;
    dconf.enable = true;
    nh.enable = true;
  };
}
