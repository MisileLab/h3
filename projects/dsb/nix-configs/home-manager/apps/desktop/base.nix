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
      imagemagick virt-manager gimp onionshare-gui appflowy firefoxpwa xfce.thunar
      /* exodus galaxy-buds-client*/ ferium prismlauncher qemu
      /* briar-desktop */ telegram-desktop
    ] ++ ([briar-desktop exodus]);
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
      extraPortals = with stablep; [xdg-desktop-portal-gtk xdg-desktop-portal-wlr];
      config.common.default = ["gtk" "wlr"];
    };
  };
  i18n = {
    inputMethod = {
      enabled = "kime";
      kime.extraConfig = ''
      daemon:
        modules:
          - Xim
          - Wayland
      engine:
        translation_layer: null
        default_category: Latin
        global_category_state: false
        global_hotkeys:
          M-C-Backslash:
            behavior: !Mode Math
            result: ConsumeIfProcessed
          S-Space:
            behavior: !Toggle
            - Hangul
            - Latin
            result: Consume
          M-C-E:
            behavior: !Mode Emoji
            result: ConsumeIfProcessed
          Esc:
            behavior: !Switch Latin
            result: Bypass
          Muhenkan:
            behavior: !Toggle
            - Hangul
            - Latin
            result: Consume
          AltR:
            behavior: !Toggle
            - Hangul
            - Latin
            result: Consume
          Hangul:
            behavior: !Toggle
            - Hangul
            - Latin
            result: Consume
        category_hotkeys:
          Hangul:
            ControlR:
              behavior: !Mode Hanja
              result: Consume
            HangulHanja:
              behavior: !Mode Hanja
              result: Consume
            F9:
              behavior: !Mode Hanja
              result: ConsumeIfProcessed
        mode_hotkeys:
          Math:
            Enter:
              behavior: Commit
              result: ConsumeIfProcessed
            Tab:
              behavior: Commit
              result: ConsumeIfProcessed
          Hanja:
            Enter:
              behavior: Commit
              result: ConsumeIfProcessed
            Tab:
              behavior: Commit
              result: ConsumeIfProcessed
          Emoji:
            Enter:
              behavior: Commit
              result: ConsumeIfProcessed
            Tab:
              behavior: Commit
              result: ConsumeIfProcessed
        candidate_font: Noto Sans CJK KR
        xim_preedit_font:
        - Noto Sans CJK KR
        - 15.0
        latin:
          layout: Qwerty
          preferred_direct: true
        hangul:
          layout: dubeolsik
          word_commit: false
          preedit_johab: Needed
          addons:
            all:
            - ComposeChoseongSsang
            dubeolsik:
            - TreatJongseongAsChoseong
      '';
    };
  };
}
