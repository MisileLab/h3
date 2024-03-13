{ config, pkgs, catppuccin, ... }:
let
  c = import ./config.nix;
  # electron-waylandify
  ewl = binaryPath: ''
    exec ${binaryPath} --enable-features=UseOzonePlatform --ozone-platform=wayland
  '';
in
{
  home.username = "misile";
  home.homeDirectory = "/home/misile";
  home.stateVersion = "23.11"; # dont change it

  home.packages = with pkgs; [
    # System
    sbctl bluez brightnessctl gnupg nix-tree cryptsetup smartmontools

    # Development
    niv cabal-install pkg-config edgedb fh nixpkgs-fmt
    hub poetry d2 micromamba pdm mypy dvc snyk ghidra pwndbg
    cargo-update pre-commit pijul

    # Some cryptos
    solana-validator

    # Language compiler and lsp
    ghc
    rustup
    python312Full
    nasm
    tailwindcss-language-server

    # Utils
    file wget imagemagick usbutils axel onefetch fastfetch ouch wgetpaste
    hyperfine hdparm duperemove hydra-check glow virt-manager
    killall delta qemu screen termscp rhash nvtop-amd genact convmv

    # Network
    dhcpcd cloudflare-warp trayscale tor-browser-bundle-bin bruno

    # Fonts
    fira-code-nerdfont nanum pretendard noto-fonts-color-emoji
    noto-fonts noto-fonts-cjk

    # Sound
    pulsemixer galaxy-buds-client

    # Some chat and game
    ferium vesktop

    # Compatibility
    figma-linux wineWowPackages.stable appimage-run
    (pkgs.writeShellScriptBin "discord" (ewl "${pkgs.vesktop}/bin/vesktop"))
    (pkgs.writeShellScriptBin "vscode" (ewl "${pkgs.vscodium}/bin/codium"))
    (pkgs.writeShellScriptBin "gdb" (ewl "${pkgs.pwndbg}/bin/pwndbg"))
    (pkgs.writeShellScriptBin "tetrio" (ewl "${pkgs.tetrio-desktop.override{withTetrioPlus=true;}}/bin/tetrio-desktop"))
    (pkgs.writeShellScriptBin "insomnia" (ewl "${pkgs.bruno}/bin/bruno"))
  ]
  ++ (with llvmPackages_latest; [libcxxClang openmp libunwind]) # llvm
  ++ (with nodePackages_latest; [nodejs pnpm typescript-language-server]) # nodejs
  ++ (with python311Packages; [pip virtualenv pipx]); # python thing

  home.file = {
    ".local/share/rofi/themes/catppuccin-mocha.rasi".source = config.lib.file.mkOutOfStoreSymlink "${builtins.fetchGit{
      url="https://github.com/catppuccin/rofi";
      rev="5350da41a11814f950c3354f090b90d4674a95ce";
    }}/basic/.local/share/rofi/themes/catppuccin-mocha.rasi";
    "non-nixos-things/catppuccin-ghidra".source = config.lib.file.mkOutOfStoreSymlink "${builtins.fetchGit{
      url="https://github.com/StanlsSlav/ghidra";
      rev="f783b5e15836964e720371c0da81819577dd2614";
    }}";
    ".config/obs-studio/themes/Catppuccin Mocha.qss".source = config.lib.file.mkOutOfStoreSymlink "${builtins.fetchGit {
      url="https://github.com/catppuccin/obs";
      rev="9a78d89d186afbdcc719a1cb7bbf7fb1c2fdd248";
    }}";
  };

  home.sessionVariables = {
    QT_QPA_PLATFORM = "wayland";
    EDITOR = "hx";
  };

  catppuccin.flavour = "mocha";
  fonts.fontconfig.enable = true;
  nixpkgs.config.allowUnfree = true;
  xdg = {
    enable = true;
    portal = {
      enable = true;
      extraPortals = with pkgs; [xdg-desktop-portal-gtk xdg-desktop-portal-wlr];
      config.common.default = ["gtk" "wlr"];
    };
  };
  gtk = {enable = true;catppuccin.enable = true;};
  programs = {
    mpv.enable = true;
    obs-studio.enable = true;
    java={enable=true;package=pkgs.temurin-bin-21;};
    go.enable = true;
    zoxide.enable = true;
    fzf.enable = true;
    tealdeer.enable = true;
    topgrade.enable = true;
    ripgrep.enable = true;
    irssi.enable = true;
    lazygit = {
      enable = true;
      catppuccin.enable = true;
      settings = {
        git.commit.signOff = true;
      };
    };
    vscode = {
      enable = true;
      package = pkgs.vscodium;
    };
    direnv = {
      enable = true;
      nix-direnv.enable = true;
    };
    eza.enable = true;
    bat = {
      enable = true;
      catppuccin.enable = true;
    };
    btop = {
      enable = true;
      catppuccin.enable = true;
    };
    fish = {
      enable = true;
      shellInit = with pkgs; ''
        fish_add_path -m ~/.cargo/bin
        fish_add_path -m ~/.avm/bin
        fish_add_path -m ~/.local/share/solana/install/active_release/bin
        
        alias nix-clean="nix store optimise && sudo nix store optimise && nix-collect-garbage -d && sudo nix-collect-garbage -d"
        alias cat="${bat}/bin/bat"
        alias ocat="${coreutils}/bin/cat"
        alias ls="${eza}/bin/eza --icons"
        alias onefetch="${onefetch}/bin/onefetch --number-of-languages 10000"
        alias cd="${zoxide}/bin/zoxide"
        
        function fzfp
          if set -q argv[1]
            $argv (${fzf}/bin/fzf --preview 'bat --color=always --style=numbers --line-range :500 {}')
          else
            ${fzf}/bin/fzf --preview 'bat --color=always --style=numbers --line-range :500 {}'
          end
        end
        function git-bulk-pulls
          if not set -q argv[1]
            set args .
          else
            set args $argv
          end
          for j in $args
            for i in $j/*
              cd $i
              ${git}/bin/git pull
              cd -
            end
          end
        end
      '';
    };
    atuin = {
      enable = true;
      enableFishIntegration = true;
    };
    git = {
      enable = true;
      lfs.enable = true;
      signing = {key = "138AC61AE9D8D2D55EAE4995CD896843C0CB9E63";signByDefault=true;};
      userName = "misilelab";
      userEmail = "misileminecord@gmail.com";
      extraConfig = { pull = {rebase = false; };
      safe = { directory = "*"; };
      init = {defaultBranch = "main";};
      delta.enable = true;
      };
    };
  };  

  # Let Home Manager install and manage itself.
  programs.home-manager.enable = true;
}
