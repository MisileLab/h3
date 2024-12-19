{pkgs, ...}:
let
  writeScript = name: content: pkgs.writeShellScriptBin name "#!${pkgs.nushell}/bin/nu\n${content} $@";
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
      portablemc miniserve openssl transmission glances lunarvim pandoc wkhtmltopdf
      yt-dlp age magic-wormhole ansifilter b3sum (emacsp) git-crypt
      aspell aspellDicts.en
      (writeScript "manual" ''
        ${pkgs.glow}/bin/glow -p ~/.config/home-manager/manual.md
      '')
      (writeScript "nix-clean" "nix-collect-garbage -d && sudo nix-collect-garbage -d && sudo nix store optimise && nix store optimise && nix-collect-garbage -d && sudo nix-collect-garbage -d")
      (writeScript "cat" "${bat}/bin/bat")
      (writeScript "ocat" "${coreutils}/bin/cat")
      (writeScript "lzg" "${lazygit}/bin/lazygit")
      (writeScript "nv" "${lunarvim}/bin/lvim")
      (writeScript "lv" "${lunarvim}/bin/lvim")
      (writeScript "doom" "~/.config/emacs/bin/doom")
      (writeScript "es" "${emacsp}/bin/emacsclient")
      (writeScript "esd" "${emacsp}/bin/emacs")
      (writeScript "git-c" "~/repos/h3/projects/dsb/utils/.venv/bin/python ~/repos/h3/projects/dsb/utils/gen-commit-message.py")
      (writeScript "utils" "~/repos/h3/projects/dsb/utils/zig-out/bin/utils")
    ];
    programs = {
      aerc = {
        enable = true;
        catppuccin.enable = true;
      };
      # nix-index.enable = true;
      glamour.catppuccin.enable = true;
      gpg = {
        enable = true;
        mutableTrust = true;
      };
      bash.enable = true;
      fzf = {
        enable = true;
        catppuccin.enable = true;
      };
      topgrade.enable = true;
      oh-my-posh = {
        enable = true;
        useTheme = "catppuccin_mocha";
      };
      nushell = {
        enable = true;
        shellAliases = {
          onefetch = "${pkgs.onefetch}/bin/onefetch --number-of-languages 9999";
          ez = "${pkgs.eza}/bin/eza --icons";
          cat = "${pkgs.bat}/bin/bat";
          ocat = "${pkgs.coreutils}/bin/cat";
          ssh = "${pkgs.kitty}/bin/kitten ssh";
        };
        extraConfig = ''
def bulk-run [paths: list<string>, command: string, ...args] {
  mut output = {}
  for $dir in $paths {
    for $p in (ls $dir | where type == dir) {
      let path = $p | get name
      print $path
      cd $path
      let result = (^$command ...$args)
      $output = ($output | upsert $path $result)
      cd -
    }
  }
  $output
}
$env.PATH = ($env.PATH | split row (char esep) | append "~/.cargo/bin")
if $env.TERM == "linux" {
  sway
}
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
