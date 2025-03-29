{pkgs, stablep, config, ...}:
{
  home = {
    sessionVariables = {
      LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
    };
    packages = with pkgs; [
      neovim wakatime

      # Development
      gel d2 pre-commit just mongosh tabnine
      snyk radicle-node infisical pnpm poop binsider

      # Language tools
      vscode-langservers-extracted
      ruby_3_4 rubyPackages_3_4.ruby-lsp
      ghc cabal-install haskell-language-server
      cargo-update rustc cargo clippy rust-analyzer
      metals scala-next
      taplo
      yaml-language-server
      kotlin kotlin-language-server android-studio gradle
      go gopls
      python313Full uv mypy
      nasm
      deno bun
      # https://github.com/NixOS/nixpkgs/issues/389150
      /*hvm bend*/
      clang-tools lldb pkg-config
      niv nixpkgs-fmt nix-tree hub fh nixfmt-rfc-style nurl
      packwiz ccemux
      unityhub dotnet-sdk_8
      lua (writeShellScriptBin "luajit" "${luajit}/bin/lua") luarocks
      vala
      zig

      # lsp
      basedpyright nil (stablep.vala-language-server) bash-language-server
      tailwindcss-language-server astro-language-server ruff lua-language-server
      marksman zls
    ]
    ++ (with llvmPackages_latest; [libcxxClang openmp libunwind]) # llvm
    ++ (with nodePackages_latest; [nodejs typescript typescript-language-server svelte-language-server prettier]) # nodejs
    ++ (with python313Packages; [pip virtualenv mitmproxy]); # python thing
    file = {
      ".config/process-compose/theme.yaml".source = config.lib.file.mkOutOfStoreSymlink "${builtins.fetchGit {
        url="https://github.com/catppuccin/process-compose";
        rev="b0c48aa07244a8ed6a7d339a9b9265a3b561464d";
      }}/themes/catppuccin-mocha.yaml";
      "/home/misile/non-nixos-things/template".text = with config.programs.git; "\nSigned-off-by: ${userName} <${userEmail}>";
    };
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
        tag = {
          gpgSign = true;
          forceSignAnnotated = true;
        };
        pull.rebase = false;
        safe.directory = "*";
        init.defaultBranch = "main";
        core.editor = "nvim";
        delta.enable = true;
        push.autoSetupRemote = true;
        commit.template = "/home/misile/non-nixos-things/template";
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
