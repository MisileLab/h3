{c, config, pkgs, stablep, ...}:
let
  briar-desktop = pkgs.callPackage ./briar.nix {};
  exodus = pkgs.callPackage ./exodus.nix {};
in
{
  imports = [
    ./security.nix
    ./electrons.nix
    ./fonts.nix
    ./compatibility.nix
    ./sway.nix
  ];
  home = {
    # https://github.com/NixOS/nixpkgs/issues/313548
    # https://github.com/NixOS/nixpkgs/issues/306670
    packages = with pkgs; [
      brightnessctl clipman wl-clipboard pavucontrol
      imagemagick virt-manager gimp appflowy xfce.thunar
      /* galaxy-buds-client */ ferium (prismlauncher.override{withWaylandGLFW=true;})
      seahorse kdePackages.filelight qemu firefoxpwa
      onionshare zed-editor jetbrains.idea-community-bin
    ] ++ ([briar-desktop exodus]) ++ (with stablep; [telegram-desktop]);
    file = {
      ".local/share/PrismLauncher/themes/catppuccin-mocha.zip".source = config.lib.file.mkOutOfStoreSymlink "${builtins.fetchGit {
        url="https://github.com/catppuccin/prismlauncher";
        rev="2edbdf5295bc3c12c3dd53b203ab91028fce2c54";
      }}/themes/Mocha/Catppuccin-Mocha.zip";
      ".config/obs-studio/themes/Catppuccin Mocha.qss".source = config.lib.file.mkOutOfStoreSymlink "${builtins.fetchGit {
        url="https://github.com/catppuccin/obs";
        rev="e7c4fcf387415a20cb747121bc0416c4c8ae3362";
      }}";
    };
    pointerCursor = {
      name = "Adwaita";
      package = pkgs.adwaita-icon-theme;
      size = 32;
    };
  };
  programs = {
    obs-studio.enable = true;
    alacritty = {
      enable = true;
      catppuccin.enable = true;
      settings = {shell = "${pkgs.fish}/bin/fish";};
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
