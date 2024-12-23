{config, pkgs, stablep, ...}:
let
  briar-desktop = pkgs.callPackage ./briar.nix {};
in
{
  imports = [
    ./security.nix
    ./waylandify.nix
    ./fonts.nix
    ./compatibility.nix
    ./sway.nix
  ];
  # https://github.com/NixOS/nixpkgs/issues/367703
  home = {
    packages = with pkgs; [
      brightnessctl clipman wl-clipboard pavucontrol
      imagemagick (stablep.virt-manager) appflowy xfce.thunar
      galaxy-buds-client ferium prismlauncher
      seahorse kdePackages.filelight firefoxpwa gparted
      onionshare jetbrains.idea-community-bin gimp telegram-desktop
      xournalpp
    ] ++ ([briar-desktop exodus]);
    file = {
      ".local/share/PrismLauncher/themes/catppuccin-mocha.zip".source = config.lib.file.mkOutOfStoreSymlink "${builtins.fetchGit {
        url="https://github.com/catppuccin/prismlauncher";
        rev="2edbdf5295bc3c12c3dd53b203ab91028fce2c54";
      }}/themes/Mocha/Catppuccin-Mocha.zip";
    };
    pointerCursor = {
      name = "Adwaita";
      package = pkgs.adwaita-icon-theme;
      size = 32;
    };
  };
  catppuccin = {
    obs.enable = true;
    kitty.enable = true;
    zed.enable = true;
  };
  programs = {
    zed-editor.enable = true;
    obs-studio = {
      enable = true;
      package = stablep.obs-studio;
    };
    kitty = {
      enable = true;
      settings.shell = "${pkgs.nushell}/bin/nu";
      font = {
        name = "FiraCode Nerd Font Mono";
        size = 11.25;
      };
      keybindings = {
        "ctrl+shift+plus" = "change_font_size all +2.0";
        "ctrl+shift+minus" = "change_font_size all -2.0";
      };
    };
    firefox = {
      enable = true;
      nativeMessagingHosts = [pkgs.firefoxpwa];
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
