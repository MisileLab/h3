{pkgs, config, ...}:
{
  home = {
    sessionVariables = {
      LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
    };
    packages = with pkgs; [
      # Development
      edgedb d2 pre-commit pijul just dive dvc solana-validator
      # https://github.com/NixOS/nixpkgs/pull/311156
      snyk ghidra /*pwndbg*/ bruno radicle-node infisical pnpm_9
      #(pkgs.writeShellScriptBin "gdb" "${pkgs.pwndbg}/bin/pwndbg")

      # Language tools
      ghc cabal-install
      rustup cargo-update
      python312Full micromamba pdm mypy ruff-lsp
      nasm
      tailwindcss-language-server volta deno
      hvm kind2
      clang-tools lldb pkg-config
      niv fh nixpkgs-fmt nix-tree hub
      marksman

      # custom file
      (pkgs.writeShellScriptBin "bs" "infisical run --project-config-dir=/home/misile/repos/h3/projects/dsb/utils -- pdm run -p ~/repos/h3/projects/dsb/utils ~/repos/h3/projects/dsb/utils/butter-shell.py")
    ]
    ++ (with llvmPackages_latest; [libcxxClang openmp libunwind]) # llvm
    ++ (with nodePackages_latest; [nodejs typescript-language-server svelte-language-server]) # nodejs
    ++ (with python312Packages; [pip virtualenv keyring keyrings-cryptfile python-lsp-server mitmproxy]); # python thing
    file = {
      "non-nixos-things/catppuccin-ghidra".source = config.lib.file.mkOutOfStoreSymlink "${builtins.fetchGit{
        url="https://github.com/StanlsSlav/ghidra";
        rev="c7e5781c3485912f49c7e4ebf469bb474ffd7d62";
      }}";
    };
  };
  programs = {
    java={enable=true;package=pkgs.temurin-bin-21;};
    go.enable = true;
    lazygit = {
      enable = true;
      catppuccin.enable = true;
      settings = {
        git.commit.signOff = true;
      };
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
