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
    topgrade sbctl tealdeer synology-drive-client bluez

    # Development
    git niv ghc cabal-install rustup pwndbg go temurin-bin-21 wineWowPackages.stable

    # Network
    dhcpcd cloudflare-warp trayscale

    # Fonts
    fira-code-nerdfont nanum openmoji-color
    
    # Sound
    pulsemixer

    # Chat
    irssi vesktop
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
    fish = {
      enable = true;
      shellInit = ''
      alias nix-clean="nix store optimise && sudo nix store optimise && nix-collect-garbage -d && sudo nix-collect-garbage -d"
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
