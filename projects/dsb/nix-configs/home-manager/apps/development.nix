{pkgs, config, ...}:
{
  home = {
    sessionVariables = {
      LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
    };
    packages = with pkgs; [
      neovim wakatime

      # Development
      gel d2 pre-commit just tabnine
      snyk radicle-node infisical poop
      gh opencode process-compose code-cursor
      netlify-cli cursor-cli tree-sitter amp-cli

      # Language tools
      pnpm yarn-berry
      dart
      vscode-langservers-extracted
      ruby_3_4 rubyPackages_3_4.ruby-lsp
      ghc cabal-install haskell-language-server
      cargo-update rustc cargo clippy rust-analyzer
      metals scala-next
      taplo
      yaml-language-server
      kotlin kotlin-language-server android-studio gradle
      go gopls
      python314 uv mypy pixi
      nasm
      deno bun
      hvm bend
      clang-tools lldb pkg-config
      niv nixpkgs-fmt nix-tree hub fh nixfmt-rfc-style nurl
      packwiz ccemux
      dotnet-sdk_9 omnisharp-roslyn
      lua (writeShellScriptBin "luajit" "${luajit}/bin/lua") luarocks
      vala
      zig

      # lsp
      basedpyright nil vala-language-server bash-language-server
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
      "/home/misile/non-nixos-things/template".text = with config.programs.git.settings.user; "\nSigned-off-by: ${name} <${email}>";
    };
  };
  programs = {
    delta = {
      enable = true;
      enableGitIntegration = true;
    };
    java = {
      enable = true;
      package = pkgs.temurin-bin-21;
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
      settings = {
        user = {
          name = "misilelab";
          email = "misileminecord@gmail.com";
        };
        tag = {
          gpgSign = true;
          forceSignAnnotated = true;
        };
        pull.rebase = false;
        safe.directory = "*";
        init.defaultBranch = "main";
        core.editor = "nvim";
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
