{pkgs, stablep, ...}:
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
      (stablep.galaxy-buds-client) kdePackages.filelight (stablep.firefoxpwa) gparted
      gimp telegram-desktop xournalpp zotero headsetcontrol
    ] ++ (with pkgs.kdePackages; [okular merkuro]);
    pointerCursor = {
      name = "Adwaita";
      package = pkgs.adwaita-icon-theme;
      size = 32;
    };
  };
  programs = {
    obs-studio.enable = true;
    ghostty = {
      enable = true;
      settings = {
        command = "${pkgs.nushell}/bin/nu";
        font-family = "FiraCode Nerd Font Mono";
        font-size = 12;
      };
    };
    firefox = {
      enable = true;
      package = stablep.firefox;
      nativeMessagingHosts = with stablep; [firefoxpwa];
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
