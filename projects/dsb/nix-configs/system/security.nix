{pkgs, ...}:
{
  imports = [
    ./applications/yubikey.nix
  ];

  programs.wireshark = {
    enable = true;
    package = pkgs.wireshark;
    usbmon.enable = true;
    dumpcap.enable = true;
  };
  users.groups.wireshark.members = [ "misile" ];

  services = {
    # fprintd.enable = true;
    clamav = {
      daemon.enable = true;
      updater.enable = true;
    };
    tor = {
      enable = true;
      client.enable = true;
    };
  };

  networking = {
    firewall.enable = false;
    wireguard.enable = true;
  };

  systemd = {
    user = {
      services.polkit-gnome-authentication-agent-1 = {
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
  };

  #environment.systemPackages = with pkgs; [fprintd];
}
