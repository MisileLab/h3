{lib, config, pkgs, zigpkgs, ...}:
let
  writeScript = name: content: pkgs.writeShellScriptBin name "${content} $@";
  completions = [
    "dotnet" "docker" "gh" "git" "glow" "just" "less" "man"
    "nano" "nix" "npm" "pnpm" "pre-commit" "rustup" "rg" "ssh"
    "tar" "vscode" "zellij" "curl" "bat" "cargo" "uv" "curl"
    "dotnet" "gradlew" "less" "make" "man" "nano" "pytest"
    "tar" "typst"
  ];
  aliases = ["docker" "eza"];
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
        yt-dlp magic-wormhole ansifilter b3sum git-crypt inxi age sops
        distrobox vulnix zap zoom-us
        (writeScript "manual" ''
          ${pkgs.glow}/bin/glow -p ~/.config/home-manager/manual.md
        '')
        (writeScript "nix-clean" "nix-collect-garbage -d && sudo nix-collect-garbage -d && sudo nix store optimise")
        (writeScript "cat" "${bat}/bin/bat")
        (writeScript "ocat" "${coreutils}/bin/cat")
        (writeScript "lzg" "${lazygit}/bin/lazygit")
        (writeScript "util" "~/repos/h3/projects/dsb/utils/zig-out/bin/utils")
        (writeScript "zig-beta" "${zigpkgs.master}/bin/zig")
      ];
    };
    sops = {
      age.keyFile = "/home/misile/.config/sops/age/keys.txt";
      defaultSopsFile = ../secrets.yaml;
      defaultSecretsMountPoint = "/run/user/1000/secrets.d";
      secrets = {
        uv_pypi_token.path = "${config.sops.defaultSymlinkPath}/uv_pypi_token";
        tavily_api_key.path = "${config.sops.defaultSymlinkPath}/tavily_api_key";
        openai_api_key.path = "${config.sops.defaultSymlinkPath}/openai_api_key";
        openrouter_api_key.path = "${config.sops.defaultSymlinkPath}/openrouter_api_key";
        ollama_api_key.path = "${config.sops.defaultSymlinkPath}/ollama_api_key";
      };
    };
    programs = {
      jrnl.enable = true;
      nix-search-tv = {
        enable = true;
        enableTelevisionIntegration = true;
      };
      television.enable = true;
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
        };
        extraConfig = ''
# plugins
plugin add ${pkgs.nushellPlugins.polars}/bin/nu_plugin_polars
plugin use polars
# completion that command name and program different
source ${pkgs.nu_scripts}/share/nu_scripts/custom-completions/tealdeer/tldr-completions.nu
source ${pkgs.nu_scripts}/share/nu_scripts/custom-completions/bend/bend-completion.nu
source ${pkgs.nu_scripts}/share/nu_scripts/custom-completions/yarn/yarn-v4-completions.nu
${lib.concatStringsSep "\n" (map (name: "source ${pkgs.nu_scripts}/share/nu_scripts/custom-completions/${name}/${name}-completions.nu") completions)}
${lib.concatStringsSep "\n" (map (name: "source ${pkgs.nu_scripts}/share/nu_scripts/aliases/${name}/${name}-aliases.nu") aliases)}
use std/util "path add"
$env.UV_PUBLISH_TOKEN = (cat ${config.sops.secrets.uv_pypi_token.path})
$env.TAVILY_API_KEY = (cat ${config.sops.secrets.tavily_api_key.path})
$env.AVANTE_OPENAI_API_KEY = (cat ${config.sops.secrets.openai_api_key.path})
$env.AVANTE_OPENROUTER_API_KEY = (cat ${config.sops.secrets.openrouter_api_key.path})
$env.AVANTE_OLLAMA_API_KEY = (cat ${config.sops.secrets.ollama_api_key.path})
$env.PNPM_HOME = "/home/misile/.local/share/pnpm/global/5"
$env.DEVSHELL_NO_MOTD = 1;
$env.EDITOR = "nvim";
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
path add $env.PNPM_HOME
if $env.TERM == "linux" {
  niri --session
}
        '';
      };
    };
    programs = {
      fabric-ai.enable = true;
    };
    services = {
      pueue.enable = true;
      gpg-agent = {
        enable = true;
        enableSshSupport = true;
        pinentry.package = pkgs.pinentry.curses;
      };
    };
  }
