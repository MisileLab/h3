{pkgs, stablep, config, ...}:
{
  home = {
    sessionVariables = {
      LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
    };
    packages = with pkgs; [
      # Development
      edgedb d2 pre-commit pijul darcs just dive (dvc.override{enableAWS=true;}) solana-cli
      (stablep.snyk) pwndbg radicle-node infisical pnpm_9 jetbrains-toolbox ghidra poop binsider
      (writeShellScriptBin "gdb" "${pwndbg}/bin/pwndbg") process-compose wakatime

      # Language tools
      ghc cabal-install
      rustup cargo-update
      python313Full uv mypy ruff
      nasm
      deno
      hvm bend
      clang-tools lldb pkg-config
      niv nixpkgs-fmt nix-tree hub fh
      marksman
      packwiz
      (stablep.unityhub) dotnet-sdk_8
      ccemux lua (writeShellScriptBin "luajit" "${luajit}/bin/lua")
      vala

      # normalnvim
      yazi grcov gnumake

      # lsp
      shellcheck basedpyright nil vala-language-server bash-language-server
      tailwindcss-language-server astro-language-server ruff lua-language-server
    ]
    ++ (with llvmPackages_latest; [libcxxClang openmp libunwind]) # llvm
    ++ (with nodePackages_latest; [nodejs typescript typescript-language-server svelte-language-server yarn]) # nodejs
    # https://nixpk.gs/pr-tracker.html?pr=355071
    ++ (with python312Packages; [pip virtualenv python-lsp-server mitmproxy]); # python thing
    file = {
      "non-nixos-things/catppuccin-ghidra".source = config.lib.file.mkOutOfStoreSymlink "${builtins.fetchGit{
        url="https://github.com/catppuccin/ghidra";
        rev="bed0999f96ee9869ed25e0f1439bef5eff341e22";
      }}";
      ".config/process-compose/theme.yaml".source = config.lib.file.mkOutOfStoreSymlink "${builtins.fetchGit {
        url="https://github.com/catppuccin/process-compose";
        rev="b0c48aa07244a8ed6a7d339a9b9265a3b561464d";
      }}/themes/catppuccin-mocha.yaml";
    };
  };
  catppuccin.lazygit.enable = true;
  programs = {
    java={enable=true;package=pkgs.temurin-bin-21;};
    go.enable = true;
    lazygit = {
      enable = true;
      settings.git.commit.signOff = true;
    };
    git = {
      enable = true;
      lfs.enable = true;
      signing = {key = "138AC61AE9D8D2D55EAE4995CD896843C0CB9E63";signByDefault=true;};
      userName = "misilelab";
      userEmail = "misileminecord@gmail.com";
      extraConfig = {
        pull.rebase = false;
        safe.directory = "*";
        init.defaultBranch = "main";
        core.editor = "lvim";
        delta.enable = true;
      };
    };
    direnv = {
      enable = true;
      nix-direnv.enable = true;
    };
  };
}
