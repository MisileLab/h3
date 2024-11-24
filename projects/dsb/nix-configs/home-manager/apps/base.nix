{pkgs, ...}:
let
  writeScript = name: content: pkgs.writeShellScriptBin name "#!${pkgs.fish}/bin/fish\n${content} $@";
  # https://github.com/NixOS/nixpkgs/pull/357119
  lunarvimp = pkgs.callPackage ./lunarvim.nix {};
  emacsp = (pkgs.emacsPackagesFor pkgs.emacs-nox).emacsWithPackages (
    epkgs: with epkgs; [
      (treesit-grammars.with-grammars (p: builtins.attrValues p))
      treesit-auto
    ]
  );
in
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
      portablemc miniserve openssl transmission glances (lunarvimp) pandoc wkhtmltopdf
      yt-dlp age magic-wormhole ansifilter b3sum (emacsp)
      (writeScript "manual" ''
        ${pkgs.glow}/bin/glow -p ~/.config/home-manager/manual.md
      '')
      (writeScript "nix-clean" "nix-collect-garbage -d && sudo nix-collect-garbage -d && sudo nix store optimise && nix store optimise && nix-collect-garbage -d && sudo nix-collect-garbage -d")
      (writeScript "cat" "${bat}/bin/bat")
      (writeScript "ocat" "${coreutils}/bin/cat")
      (writeScript "lzg" "${lazygit}/bin/lazygit")
      (writeScript "nv" "${lunarvimp}/bin/lvim")
      (writeScript "lv" "${lunarvimp}/bin/lvim")
      (writeScript "doom" "~/.config/emacs/bin/doom")
      (writeScript "es" "${emacsp}/bin/emacsclient")
      (writeScript "esd" "${emacsp}/bin/emacs")
      (writeScript "git-c" "~/repos/h3/projects/dsb/utils/.venv/bin/python ~/repos/h3/projects/dsb/utils/gen-commit-message.py")
    ];
    programs = {
      aerc = {
        enable = true;
        catppuccin.enable = true;
      };
      nushell.enable = true;
      # nix-index.enable = true;
      glamour.catppuccin.enable = true;
      gpg = {
        enable = true;
        mutableTrust = true;
      };
      bash = {
        enable = true;
        initExtra = ''
          if [[ $(tty) == "/dev/tty1" ]]; then
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
        shellAliases = {
          onefetch = "${pkgs.onefetch}/bin/onefetch --number-of-languages 9999";
          ls = "${pkgs.eza}/bin/eza --icons";
          cat = "${pkgs.bat}/bin/bat";
          ocat = "${pkgs.coreutils}/bin/cat";
        };
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
          function git-bulk-status
            if not set -q argv[1]
              set args
            else
              set args $argv
            end
            for j in $args
              for i in $j/*
                cd $i
                echo $i
                set output (command git status --porcelain=v2)
                if not test -z "$output"
                  ${git}/bin/git status
                end
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
      emacs = {
        enable = true;
        package = emacsp;
      };
      gpg-agent = {
        enable = true;
        enableSshSupport = true;
        pinentryPackage = pkgs.pinentry.curses;
      };
    };
  }
