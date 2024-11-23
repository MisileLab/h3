{pkgs, stablep, config, ...}:
{
  home = {
    sessionVariables = {
      LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
    };
    packages = with pkgs; [
      # Development
      edgedb d2 pre-commit pijul darcs just dive (dvc.override{enableAWS=true;}) solana-validator
      snyk pwndbg radicle-node infisical pnpm_9 jetbrains-toolbox ghidra poop binsider
      (pkgs.writeShellScriptBin "gdb" "${pkgs.pwndbg}/bin/pwndbg") process-compose

      # Language tools
      ghc cabal-install
      rustup cargo-update
      python313Full uv mypy ruff-lsp
      nasm
      tailwindcss-language-server deno astro-language-server
      hvm bend
      clang-tools lldb pkg-config
      niv nixpkgs-fmt nix-tree hub fh
      marksman
      packwiz
      unityhub dotnet-sdk_8
      ccemux lua-language-server lua (pkgs.writeShellScriptBin "luajit" "${pkgs.luajit}/bin/lua")

      # lsp
      shellcheck basedpyright

      # custom file
      (pkgs.writeShellScriptBin "bs" "infisical run --project-config-dir=/home/misile/repos/h3/projects/dsb/utils -- ${pkgs.uv}/bin/uv run -p ~/repos/h3/projects/dsb/utils ~/repos/h3/projects/dsb/utils/butter-shell.py")
    ]
    ++ (with stablep.llvmPackages_latest; [libcxxClang openmp libunwind]) # llvm
    # https://github.com/NixOS/nixpkgs/pull/356257
    ++ (with nodePackages_latest; [(pkgs.nodejs_22) typescript-language-server svelte-language-server]) # nodejs
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
  programs = {
    java={enable=true;package=pkgs.temurin-bin-21;};
    go.enable = true;
    lazygit = {
      enable = true;
      catppuccin.enable = true;
      settings.git.commit.signOff = true;
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
    direnv = {
      enable = true;
      nix-direnv.enable = true;
    };
  };
}
