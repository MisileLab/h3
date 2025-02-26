{lib, pkgs, zigpkgs, secrets, ...}:
let
  writeScript = name: content: pkgs.writeShellScriptBin name "${content} $@";
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
          ssh = "kitten ssh";
        };
        extraConfig = ''
# completion that command name and program different
source ${pkgs.nu_scripts}/share/nu_scripts/custom-completions/tealdeer/tldr-completions.nu
${lib.concatStringsSep "\n" (map (name: "source ${pkgs.nu_scripts}/share/nu_scripts/custom-completions/${name}/${name}-completions.nu") completions)}
${lib.concatStringsSep "\n" (map (name: "source ${pkgs.nu_scripts}/share/nu_scripts/aliases/${name}/${name}-aliases.nu") aliases)}
# custom completions
source ~/non-nixos-things/scripts/uv-completions.nu
# https://github.com/nushell/nu_scripts/blob/main/custom-completions/ssh/ssh-completions.nu
extern "kitten ssh" [
  destination?: string@"nu-complete ssh-host"
  -4            # Forces ssh to use IPv4 addresses only.
  -6            # Forces ssh to use IPv6 addresses only.
  -A            # Enables forwarding of connections from an authentication agent such as ssh-agent(1).
  -a            # Disables forwarding of the authentication agent connection.
  -B: string    # bind_interface
  -b: string    # bind_address
  -C            # Requests compression of all data
  -c: string    # cipher_spec
  -D            # [bind_address:]port
  -E: string    # log_file
  -e            # escape_char
  -F: string    # configfile
  -f            # Requests ssh to go to background just before command execution.
  -G            # Causes ssh to print its configuration after evaluating Host and Match blocks and exit.
  -g            # Allows remote hosts to connect to local forwarded ports
  -I: string    # pkcs11
  -i: string    # identity_file
  -J: string    # destination
  -K            # Enables GSSAPI-based authentication and forwarding(delegation) of GSSAPI credentials to the server.
  -k            # Disables forwarding (delegation) of GSSAPI credentials to the server.
  -L: string    # [bind_address:]port:host:hostport / [bind_address:]port:remote_socket / local_socket:host:hostport / local_socket:remote_socket
  -l: string    # login_name
  -M            # Places the ssh client into “master” mode for connection sharing.
  -m: string    # mac_spec
  -N            # Do not execute a remote command.
  -n            # Redirects stdin from /dev/null (5) for details.
  -O: string    # ctl_cmd
  -o: string    # option
]
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
path add "~/.local/bin"
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
