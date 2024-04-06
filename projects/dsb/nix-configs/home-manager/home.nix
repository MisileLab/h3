{ config, pkgs, catppuccin, ... }:
let
  c = import ./config.nix;
  # electron-waylandify
  ewl = binaryPath: ''
    exec ${binaryPath} --enable-features=UseOzonePlatform --ozone-platform=wayland
  '';
  bvtop = pkgs.makeDesktopItem {
    name = "bvtop";
    desktopName = "bvtop";
    icon = "btop";
    exec = "${pkgs.alacritty}/bin/alacritty -e zellij --layout /home/misile/non-nixos-things/bvtop.kdl";
  };
  nurpkgs = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/master.tar.gz") { inherit pkgs; };
in
{
  home.username = "misile";
  home.homeDirectory = "/home/misile";
  home.stateVersion = "23.11"; # dont change it
  home.packages = with pkgs; [
    # System
    sbctl bluez brightnessctl nix-tree cryptsetup smartmontools
    borgbackup clipman wl-clipboard pavucontrol rclone pass-wayland

    # Development
    niv cabal-install pkg-config edgedb fh nixpkgs-fmt
    hub poetry micromamba pdm mypy dvc snyk ghidra pwndbg d2
    cargo-update pre-commit pijul just

    # Some cryptos
    solana-validator exodus

    # Language compiler and lsp
    ghc
    rustup
    python312Full
    nasm
    tailwindcss-language-server
    hvm kind2
    clang-tools lldb

    # Utils
    file wget imagemagick usbutils axel onefetch fastfetch ouch wgetpaste
    hyperfine hdparm duperemove hydra-check glow virt-manager
    killall delta qemu screen termscp rhash nvtopPackages.amd genact convmv
    bvtop dasel gimp onionshare-gui

    # Network
    dhcpcd cloudflare-warp trayscale tor-browser-bundle-bin bruno

    # Fonts
    fira-code-nerdfont nanum pretendard noto-fonts-color-emoji
    noto-fonts noto-fonts-cjk

    # Sound
    pulsemixer galaxy-buds-client

    # Some chat and game
    ferium vesktop prismlauncher telegram-desktop

    # Compatibility
    figma-linux wineWowPackages.stable appimage-run libreoffice
    (pkgs.writeShellScriptBin "figma" (ewl "${pkgs.figma-linux}/bin/figma-linux"))
    (pkgs.writeShellScriptBin "discord" (ewl "${pkgs.vesktop}/bin/vesktop"))
    (pkgs.writeShellScriptBin "vscode" (ewl "${pkgs.vscodium}/bin/codium"))
    (pkgs.writeShellScriptBin "gdb" "${pkgs.pwndbg}/bin/pwndbg")
    (pkgs.writeShellScriptBin "tetrio" (ewl "${pkgs.tetrio-desktop.override{withTetrioPlus=true;}}/bin/tetrio"))
    (pkgs.writeShellScriptBin "insomnia" (ewl "${pkgs.bruno}/bin/bruno"))
    (pkgs.writeShellScriptBin "manual" ''
      ${pkgs.glow}/bin/glow -p ~/non-nixos-things/manual.md
    '')
  ]
  ++ (with llvmPackages_latest; [libcxxClang openmp libunwind]) # llvm
  ++ (with nodePackages_latest; [nodejs pnpm typescript-language-server svelte-language-server]) # nodejs
  ++ (with python312Packages; [pip virtualenv keyring keyrings-cryptfile]); # python thing

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
    ".local/share/PrismLauncher/themes/catppuccin-mocha.zip".source = config.lib.file.mkOutOfStoreSymlink "${builtins.fetchGit {
      url="https://github.com/catppuccin/prismlauncher";
      rev="baa824d2738477ee54beb560ae992c834d43b757";
    }}/themes/Mocha/Catppuccin-Mocha.zip";
    "non-nixos-things/bvtop.kdl".text = "
      layout {
        tab {
          pane command=\"btop\"
        }
        tab {
          pane command=\"nvtop\"
        }
        tab {
          pane command=\"auto-cpufreq\" {
            args \"--stats\"
          }
        }
      }
    ";
    "non-nixos-things/manual.md".text = "
> Ultimately, arguing that you don't care about the right to privacy because you have nothing to hide is 
> no different than saying you don't care about free speech because you have nothing to say. - Edward Snowden

# Introduction
tldr for [privacyguides](http://www.xoe4vn5uwdztif6goazfbmogh6wh5jc4up35bqdflu6bkdc5cas5vjqd.onion)

# Protect the information
1. What do I want to protect?
2. Who do I want to protect it from?
3. How likely is it that I will need to protect it?
4. How bad are the consequences if I fail?
5. How much trouble am I willing to go through to try to prevent potential consequences?

# Common Threats
## Anonimity and Privacy
Privacy -> most at time, it's enough\
Anonimity -> send information that privacy isnt enough

## Untrusted code
this can tldr to one sentence.\
Do not view untrusted document (or even trusted) before read. If can't read, open document on whonix container

## Limiting public Information
Delete unused account. View bitwarden's data breach.

## Avoid censorship
Use matrix or discord with gpg for privacy, Use briar for anonimity

## Manual E2EE (Misile's addition)
1. Encrypt with temp gpg key\
2. Gives encrypted content with un-E2EE chat\
3. Gives gpg public key with pastebin alternatives website with onion

## Additional clean (Misile's addition)
1. Clean metadata when anonimity needs\
2. Use tails when anonimity needs\
3. Send on whonix when anonimity needs

# Account
## Creating Account
Use email aliasing and protonmail for privacy, use onionmail(only receive), anon send mail for anonymity\
Use oauth when site is trusted

## Deleting Account (Information)
read [this](#Limiting public Information)\

### Fake information
Use temp mail for mail register, use email name for fake name, faker(.py or .js) for generating fake information

### Delete it
[just delete from this](https://justdeleteme.xyz)

Avoid new account, question to self when make it

## Password
Just use unique, random password using random password generator

## SMS, email MFA auth
email MFA is just email auth, SMS needs to be wrapped(anonymity), so use [fake phone number (free)](https://quackr.io)

## TOTP
Just use [this](https://auth.ente.io)

# Payment
Use monero (or at least cryptocurrency) for most time.

## Dont use real card
- [use gift card with crypto](https://coincards.com)
- [credit card generator](https://buy.cakepay.com)

# Tools
I cant do tldr for tools, just go [this](http://www.xoe4vn5uwdztif6goazfbmogh6wh5jc4up35bqdflu6bkdc5cas5vjqd.onion/en/tools)
    ";
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
    glamour.catppuccin.enable=true;
    gpg = {
      enable = true;
      mutableTrust = true;
    };
    bash = {
      enable = true;
      initExtra = ''
        if [[ $(tty) == "/dev/tty1" ]] then
          sway
        fi
      '';
    };
    mpv.enable = true;
    obs-studio.enable = true;
    java={enable=true;package=pkgs.temurin-bin-21;};
    go.enable = true;
    zoxide = {
      enable = true;
      options = ["--cmd cd"];
    };
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
      shellAliases = with pkgs; {
        nix-clean = "nix store optimise && sudo nix store optimise && nix-collect-garbage -d && sudo nix-collect-garbage -d";
        cat = "${bat}/bin/bat";
        ocat = "${coreutils}/bin/cat";
        ls = "${eza}/bin/eza --icons";
        onefetch = "${onefetch}/bin/onefetch --number-of-languages 9999";
        lzg = "${lazygit}/bin/lazygit";
      };
      shellInit = with pkgs; ''
        fish_add_path -m ~/.cargo/bin
        fish_add_path -m ~/.avm/bin
        fish_add_path -m ~/.local/share/solana/install/active_release/bin        
        
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
      settings = {
        history_filter = [
          "ls" "clear" "cd *" "manual" "poweroff" "zellij *" "exit" "topgrade" "zellij *"
        ];
      };
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
