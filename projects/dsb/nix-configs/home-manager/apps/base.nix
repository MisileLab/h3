{lib, pkgs, zigpkgs, secrets, ...}:
let
  writeScript = name: content: pkgs.writeShellScriptBin name "#!${pkgs.nushell}/bin/nu\n${content} $@";
  completions = [
    "dotnet" "docker" "gh" "git" "glow" "just" "less" "man"
    "nano" "nix" "npm" "pnpm" "pre-commit" "rustup" "rg" "ssh"
    "tar" "vscode" "zellij" "curl" "bat" "cargo"
  ];
  aliases = ["docker" "git" "eza"];
in
  {
    imports = [
      ./development.nix
      ./network.nix
      ./monitor.nix
      ./network.nix
      ./utils.nix
    ];
    home = {
      packages = with pkgs; [
        sbctl bluez cryptsetup smartmontools borgbackup rclone pulsemixer
        portablemc miniserve openssl transmission attic-client
        yt-dlp magic-wormhole ansifilter b3sum git-crypt inxi
        (writeScript "manual" ''
          ${pkgs.glow}/bin/glow -p ~/.config/home-manager/manual.md
        '')
        (writeScript "nix-clean" "nix-collect-garbage -d && sudo nix-collect-garbage -d && sudo nix store optimise")
        (writeScript "cat" "${bat}/bin/bat")
        (writeScript "ocat" "${coreutils}/bin/cat")
        (writeScript "lzg" "${lazygit}/bin/lazygit")
        (writeScript "utils" "~/repos/h3/projects/dsb/utils/zig-out/bin/utils")
        (writeScript "zig-beta" "${zigpkgs.master}/bin/zig")
      ];
      activation = {
        installCompletions = lib.hm.dag.entryAfter ["writeBoundary"] ''
          ${pkgs.uv}/bin/uv generate-shell-completion nushell > /home/misile/non-nixos-things/scripts/uv-completions.nu
        '';
      };
    };
    programs = {
      gpg = {
        enable = true;
        mutableTrust = true;
      };
      bash.enable = true;
      fzf.enable = true;
      topgrade.enable = true;
      oh-my-posh = {
        enable = true;
        useTheme = "catppuccin_mocha";
      };
      nushell = {
        enable = true;
        shellAliases = {
          onefetch = "${pkgs.onefetch}/bin/onefetch --number-of-languages 9999";
          cat = "${pkgs.bat}/bin/bat";
          ocat = "${pkgs.coreutils}/bin/cat";
          ssh = "${pkgs.kitty}/bin/kitten ssh";
        };
        extraConfig = ''
# completion that command name and program different
source ${pkgs.nu_scripts}/share/nu_scripts/custom-completions/tealdeer/tldr-completions.nu
${lib.concatStringsSep "\n" (map (name: "source ${pkgs.nu_scripts}/share/nu_scripts/custom-completions/${name}/${name}-completions.nu") completions)}
${lib.concatStringsSep "\n" (map (name: "source ${pkgs.nu_scripts}/share/nu_scripts/aliases/${name}/${name}-aliases.nu") aliases)}
# custom completions
source ~/non-nixos-things/scripts/uv-completions.nu
use std/util "path add"
$env.UV_PUBLISH_TOKEN = "${secrets.UV_PUBLISH_TOKEN or "you_need_to_change_in_secrets_nix"}"
$env.config.hooks.command_not_found = {
  |x|
  print (command-not-found $x | str trim)
}
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
path add "~/.cargo/bin"
if $env.TERM == "linux" {
  sway
}
        '';
      };
    };
    services = {
      gpg-agent = {
        enable = true;
        enableSshSupport = true;
        pinentryPackage = pkgs.pinentry.curses;
      };
    };
  }
