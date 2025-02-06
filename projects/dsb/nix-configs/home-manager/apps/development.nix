{pkgs, stablep, config, ...}:
{
  home = {
    sessionVariables = {
      LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
    };
    packages = with pkgs; [
      neovim wakatime

      # Development
      edgedb d2 pre-commit just
      snyk radicle-node infisical pnpm_9 (stablep.poop) binsider

      # Language tools
      ghc cabal-install
      rustup cargo-update
      python313Full uv mypy
      nasm
      deno
      hvm bend
      clang-tools lldb pkg-config
      niv nixpkgs-fmt nix-tree hub fh nixfmt-rfc-style
      packwiz ccemux
      unityhub dotnet-sdk_8
      lua (writeShellScriptBin "luajit" "${luajit}/bin/lua") luarocks
      vala
      (stablep.zig)

      # lsp
      basedpyright nil vala-language-server bash-language-server
      tailwindcss-language-server astro-language-server ruff lua-language-server
      marksman (stablep.zls)
    ]
    ++ (with llvmPackages_latest; [libcxxClang openmp libunwind]) # llvm
    ++ (with nodePackages_latest; [nodejs typescript typescript-language-server svelte-language-server]) # nodejs
    ++ (with python313Packages; [pip virtualenv mitmproxy]); # python thing
    file = {
      ".config/process-compose/theme.yaml".source = config.lib.file.mkOutOfStoreSymlink "${builtins.fetchGit {
        url="https://github.com/catppuccin/process-compose";
        rev="b0c48aa07244a8ed6a7d339a9b9265a3b561464d";
      }}/themes/catppuccin-mocha.yaml";
    };
  };
  catppuccin = {
    lazygit.enable = true;
  };
  programs = {
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
        core.editor = "nvim";
        delta.enable = true;
        push.autoSetupRemote = true;
      };
    };
    direnv = {
      enable = true;
      nix-direnv.enable = true;
    };
    darcs = {
      enable = true;
      author = [ "misilelab <misile@duck.com>" ];
    };
  };
}
