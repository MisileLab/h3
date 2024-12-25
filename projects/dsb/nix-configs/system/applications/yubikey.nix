{pkgs, ...}:
{
  security = {
    rtkit.enable = true;
    pam = {
      yubico = {
        enable = true;
        mode = "challenge-response";
        id = [ "26906254" "29497348" ];
      };
      services = {
        swaylock.text = ''
          auth sufficient pam_unix.so try_first_pass likeauth nullok
          auth sufficient pam_u2f.so
          auth sufficient pam_fprintd.so
          auth include login
        '';
        login.u2fAuth = true;
        sudo.u2fAuth = true;
      };
    };
  };

  services = {
    udev = {
      packages = [ pkgs.yubikey-personalization ];
      extraRules = ''
            ACTION=="remove",\
            ENV{ID_BUS}=="usb",\
            ENV{ID_MODEL_ID}=="0407",\
            ENV{ID_VENDOR_ID}=="1050",\
            ENV{ID_VENDOR}=="Yubico",\
            RUN+="${pkgs.systemd}/bin/loginctl lock-sessions"
        '';
    };
  };
}
