{pkgs, stablep, config, ...}:
{
  home = {
    sessionVariables = {
      LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
    };
    packages = with pkgs; [
      # Development
      edgedb d2 pre-commit darcs just
      (stablep.snyk) radicle-node infisical pnpm_9 poop binsider

      # Language tools
      ghc cabal-install
      rustup cargo-update
      python313Full uv mypy ruff
      nasm
      deno
      hvm bend
      clang-tools lldb pkg-config
      niv nixpkgs-fmt nix-tree hub fh
      packwiz ccemux
      (stablep.unityhub) dotnet-sdk_8
      lua (writeShellScriptBin "luajit" "${luajit}/bin/lua")
      vala

      # lsp
      shellcheck basedpyright nil vala-language-server bash-language-server
      tailwindcss-language-server astro-language-server ruff lua-language-server
      marksman
    ]
    ++ (with llvmPackages_latest; [libcxxClang openmp libunwind]) # llvm
    ++ (with nodePackages_latest; [nodejs typescript typescript-language-server svelte-language-server yarn]) # nodejs
    # python-lsp-server failed
    ++ (with python312Packages; [pip virtualenv python-lsp-server mitmproxy]); # python thing
    file = {
      ".config/process-compose/theme.yaml".source = config.lib.file.mkOutOfStoreSymlink "${builtins.fetchGit {
        url="https://github.com/catppuccin/process-compose";
        rev="b0c48aa07244a8ed6a7d339a9b9265a3b561464d";
      }}/themes/catppuccin-mocha.yaml";
    };
  };
  catppuccin = {
    lazygit.enable = true;
    helix.enable = true;
  };
  programs = {
    helix = {
      enable = true;
    };
    java = {
      enable=true;
      package=pkgs.temurin-bin-21;
    };
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
        core.editor = "hx";
        delta.enable = true;
      };
    };
    direnv = {
      enable = true;
      nix-direnv.enable = true;
    };
  };
}
