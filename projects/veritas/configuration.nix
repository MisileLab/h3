{pkgs, ...}:
{
  imports = [
    ./disks.nix
  ];

  nix = {
    package = pkgs.lix;
    settings.experimental-features = [ "nix-command" "flakes" ];
  };
  boot = {
    loader = {
      systemd-boot.enable = true;
      efi.canTouchEfiVariables = true;
    };
    blacklistedKernelModules = [ "hyperv_fb" ];
    kernel.sysctl."net.ipv6.conf.all.forwarding" = 1;
  };

  # Enable networking
  networking = {
    hostName = "veritas"; # Define your hostname.
    networkmanager.enable = true;
    firewall = {
      allowedTCPPorts = [ 22 ];
    };
  };

  time.timeZone = "Asia/Seoul";

  # Select internationalisation properties.
  i18n = {
    defaultLocale = "en_US.UTF-8";
    extraLocaleSettings = {
      LC_ADDRESS = "en_US.UTF-8";
      LC_IDENTIFICATION = "en_US.UTF-8";
      LC_MEASUREMENT = "en_US.UTF-8";
      LC_MONETARY = "en_US.UTF-8";
      LC_NAME = "en_US.UTF-8";
      LC_NUMERIC = "en_US.UTF-8";
      LC_PAPER = "en_US.UTF-8";
      LC_TELEPHONE = "en_US.UTF-8";
      LC_TIME = "en_US.UTF-8";
    };
  };

  virtualisation.hypervGuest.enable = true;

  # Configure keymap in X11
  services = {
    openssh.enable = true;
    slunchv2.enable = true;
  };

  # Define a user account. Don't forget to set a password with ‘passwd’.
  users.users = {
    veritas = {
      isNormalUser = true;
      description = "avg blue archive enjoyer";
      extraGroups = [ "networkmanager" "wheel" ];
      shell = pkgs.fish;
    };
  };
  # List packages installed in system profile. To search, run:
  # $ nix search wget
  environment.systemPackages = with pkgs; [
    wget
  ];

  programs = {
    git.enable = true;
    fish.enable = true;
  };

  # This value determines the NixOS release from which the default
  # settings for stateful data, like file locations and database versions
  # on your system were taken. It‘s perfectly fine and recommended to leave
  # this value at the release version of the first install of this system.
  # Before changing this value read the documentation for this option
  # (e.g. man configuration.nix or on https://nixos.org/nixos/options.html).
  system.stateVersion = "24.11"; # Did you read the comment?
}
