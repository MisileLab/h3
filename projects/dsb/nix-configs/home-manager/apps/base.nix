{pkgs, ...}: 
{
  imports = [
    ./development.nix
    ./network.nix
    ./monitor.nix
    ./network.nix
    ./utils.nix
  ];
  home.packages = with pkgs; [
    sbctl bluez cryptsetup smartmontools borgbackup rclone pulsemixer
    portablemc miniserve openssl transmission glance glances lunarvim
    (pkgs.writeShellScriptBin "manual" ''
      ${pkgs.glow}/bin/glow -p ~/.config/home-manager/manual.md
    '')
    (pkgs.writeShellScriptBin "nix-clean" "nix store optimise && sudo nix store optimise && nix-collect-garbage -d && sudo nix-collect-garbage -d")
    (pkgs.writeShellScriptBin "cat" "${pkgs.bat}/bin/bat")
    (pkgs.writeShellScriptBin "ocat" "${pkgs.coreutils}/bin/cat")
    (pkgs.writeShellScriptBin "ls" "${pkgs.bat}/bin/bat")
    (pkgs.writeShellScriptBin "lzg" "${pkgs.lazygit}/bin/lazygit")
    (pkgs.writeShellScriptBin "gdiff" "${pkgs.git}/bin/diff | ${pkgs.delta}/bin/delta")
    (pkgs.writeShellScriptBin "nv" "${pkgs.lunarvim}/bin/lvim")
    (pkgs.writeShellScriptBin "lv" "${pkgs.lunarvim}/bin/lvim")
  ];
  programs = {
    # nix-index.enable = true;
    glamour.catppuccin.enable = true;
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
    fzf = {
      enable = true;
      catppuccin.enable = true;
    };
    topgrade.enable = true;
    fish = {
      enable = true;
      catppuccin.enable = true;
      shellAliases = {onefetch = "${pkgs.onefetch}/bin/onefetch --numbers-of-languages 9999";};
      plugins = [{name="tide"; src=pkgs.fishPlugins.tide.src;}];
      shellInit = with pkgs; ''
        fish_add_path -m ~/.cargo/bin
        fish_add_path -m ~/.avm/bin
        fish_add_path -m ~/.local/share/solana/install/active_release/bin
        fish_add_path -m ~/.volta/bin
        
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
    neovim = {
      enable = true;
      catppuccin.enable = true;
    };
  };
  services = {
    gpg-agent = {
      enable = true;
      enableSshSupport = true;
      extraConfig = "pinentry-program ${pkgs.pinentry.curses}/bin/pinentry-curses";
    };
  };
}
