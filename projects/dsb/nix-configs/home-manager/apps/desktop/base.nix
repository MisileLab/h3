{pkgs, ...}:
{
  imports = [
    ./security.nix
    ./waylandify.nix
    ./fonts.nix
    ./compatibility.nix
    ./sway.nix
  ];
  home = {
    packages = with pkgs; [
      brightnessctl clipman wl-clipboard pavucontrol
      imagemagick virt-manager xfce.thunar
      galaxy-buds-client kdePackages.filelight firefoxpwa gparted
      gimp telegram-desktop xournalpp
    ];
    pointerCursor = {
      name = "Adwaita";
      package = pkgs.adwaita-icon-theme;
      size = 32;
    };
  };
  catppuccin.enable = true;
  programs = {
    obs-studio.enable = true;
    kitty = {
      enable = true;
      settings.shell = "${pkgs.nushell}/bin/nu";
      font = {
        name = "FiraCode Nerd Font Mono";
        size = 11.25;
      };
      keybindings = {
        "ctrl+shift+plus" = "change_font_size all +1.0";
        "ctrl+shift+minus" = "change_font_size all -1.0";
      };
    };
    firefox = {
      enable = true;
      nativeMessagingHosts = with pkgs; [firefoxpwa];
    };
  };
  xdg = {
    enable = true;
    portal = {
      enable = true;
      extraPortals = with pkgs; [xdg-desktop-portal-gtk xdg-desktop-portal-wlr];
      config.common.default = ["gtk" "wlr"];
    };
  };
  i18n = {
    inputMethod = {
      enabled = "fcitx5";
      fcitx5.addons = with pkgs; [libsForQt5.fcitx5-qt catppuccin-fcitx5 fcitx5-gtk fcitx5-hangul];
    };
  };
}
