{c, config, pkgs, /*unstable-pkgs,*/ ...}: {
  imports = [
    ./security.nix
    ./electrons.nix
    ./fonts.nix
    ./compatibility.nix
    ./sway.nix
  ];
  home = {
    packages = with pkgs; [
      brightnessctl clipman wl-clipboard pavucontrol
      imagemagick virt-manager gimp onionshare-gui appflowy firefoxpwa
      exodus galaxy-buds-client ferium prismlauncher /*qemu telegram-desktop*/
    ];
    file = {
      ".local/share/PrismLauncher/themes/catppuccin-mocha.zip".source = config.lib.file.mkOutOfStoreSymlink "${builtins.fetchGit {
        url="https://github.com/catppuccin/prismlauncher";
        rev="07e9c3ca0ff8eb7da9fa8b5329a9d2ceec707f24";
      }}/themes/Mocha/Catppuccin-Mocha.zip";
      ".config/obs-studio/themes/Catppuccin Mocha.qss".source = config.lib.file.mkOutOfStoreSymlink "${builtins.fetchGit {
        url="https://github.com/catppuccin/obs";
        rev="9a78d89d186afbdcc719a1cb7bbf7fb1c2fdd248";
      }}";
    };
    pointerCursor = {
      name = "Adwaita";
      package = pkgs.gnome.adwaita-icon-theme;
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
    firefox.enable = true;
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
      enabled = "kime";
    };
  };
}
