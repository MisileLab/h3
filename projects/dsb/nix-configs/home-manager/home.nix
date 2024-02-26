{ config, pkgs, catppuccin, ... }:
let
  c = import ./config.nix;
in
{
  home.username = "misile";
  home.homeDirectory = "/home/misile";
  home.stateVersion = "23.11"; # dont change it

  home.packages = with pkgs; [
    # System
    topgrade sbctl tealdeer synology-drive-client bluez brightnessctl gnupg

    # Development
    git niv ghc cabal-install rustup pwndbg go temurin-bin-21
    python312Full pkg-config edgedb fh nixpkgs-fmt
    virtualenv hub poetry d2 micromamba pdm
    mypy dvc snyk

    # Utils
    file wget imagemagick usbutils axel onefetch fastfetch ouch wget-paste
    hyperfine hdparm duperemove hydra-check glow pip obs-studio

    # Network
    dhcpcd cloudflare-warp trayscale tor

    # Fonts
    fira-code-nerdfont nanum openmoji-color pretendard

    # Sound
    pulsemixer galaxy-buds-client mpv

    # Some chat and game
    irssi portablemc ferium
    (tetrio-desktop.override {
      withTetrioPlus = true;
    })

    # Compatibility
    figma-linux vesktop wineWowPackages.stable appimage-run
  ]
  ++ (with llvmPackages_latest; [clangUseLLVM openmp libunwind]) # llvm
  ++ (with nodePackages_latest; [nodejs pnpm]); # nodejs

  home.file = {
    ".local/share/rofi/themes/catppuccin-mocha.rasi".source = config.lib.file.mkOutOfStoreSymlink "${builtins.fetchGit{
      url="https://github.com/catppuccin/rofi";
      rev="5350da41a11814f950c3354f090b90d4674a95ce";
    }}/basic/.local/share/rofi/themes/catppuccin-mocha.rasi";
  };

  home.sessionVariables = {
    QT_QPA_PLATFORM = "wayland";
    EDITOR = "nvim";
  };

  catppuccin.flavour = "mocha";
  fonts.fontconfig.enable = true;
  nixpkgs.config.allowUnfree = true;
  programs = {
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
      shellInit = ''
      alias nix-clean="nix store optimise && sudo nix store optimise && nix-collect-garbage -d && sudo nix-collect-garbage -d"
      alias cat="bat"
      alias ocat="${pkgs.coreutil}/bin/cat"
      alias ls="eza --icons"
      alias onefetch="onefetch --number-of-languages 10000"
      function fzfp
        if set -q argv[1]
          $argv (${pkgs.fzf} --preview 'bat --color=always --style=numbers --line-range :500 {}')
        else
          ${pkgs.fzf} --preview 'bat --color=always --style=numbers --line-range :500 {}'
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
            git pull
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
      };
    };
  };

  #sops = {age.sshKeyPaths=["/home/misile/.ssh/id_rsa"];defaultSopsFile=/home/misile/h3/projects/dsb/nix-configs/secrets.yaml;};
  #home.activation.setupEtc = config.lib.dag.entryAfter [ "writeBoundary" ] ''
  #  /run/current-system/sw/bin/systemctl start --user sops-nix
  #'';
  #systemd.user.services.mbsync.Unit.After = [ "sops-nix.service" ];

  # Let Home Manager install and manage itself.
  programs.home-manager.enable = true;
}
