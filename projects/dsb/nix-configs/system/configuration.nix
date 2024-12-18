{ pkgs, options, ... }:
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
    locate = {
      enable = true;
      package = pkgs.plocate;
      localuser = null;
      interval = "never";
      prunePaths = options.services.locate.prunePaths.default ++ ([
        "/home/misile/.cargo"
        "/home/misile/.config/unity3d/cache"
        "/home/misile/.local/share/pnpm"
        "/home/misile/.docker"
        "/home/misile/.dotnet"
        "/home/misile/.gradle"
        "/home/misile/.tor project"
        "/home/misile/.mozilla"
        "/home/misile/.net"
        "/home/misile/.npm"
        "/home/misile/.nuget"
        "/home/misile/.radicle"
        "/home/misile/.rustup"
        "/home/misile/.vscode-oss"
        "/home/misile/.vscode"
        "/home/misile/.volta"
        "/home/misile/.var"
        "/home/misile/non-nixos-things/waydroid_script"
        "/home/misile/go"
        "/home/misile/micromamba"
        "/home/misile/Unity"
        "/home/misile/Documents"
        "/var/lib/waydroid"
        "/var/lib/flatpak"
        "/var/lib/docker"
        "/var/lib/clamav"
        "/var/log"
        "/root"
      ]);
      pruneNames = options.services.locate.pruneNames.default ++ ([
        "node_modules"
        ".wine"
        ".venv"
        ".mypy_cache"
        "media"
        "__pycache__"
        ".ruff_cache"
        ".svelte-kit"
        ".zig-cache"
        "zig-cache"
      ]);
    };
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
