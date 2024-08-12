{pkgs, config, ...}:
{
  home = {
    sessionVariables = {
      LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
    };
    packages = with pkgs; [
      # Development
      # https://github.com/NixOS/nixpkgs/issues/333739
      edgedb d2 pre-commit pijul just dive /* dvc */ solana-validator
      snyk pwndbg radicle-node infisical pnpm_9 jetbrains-toolbox ghidra
      (pkgs.writeShellScriptBin "gdb" "${pkgs.pwndbg}/bin/pwndbg")

      # Language tools
      ghc cabal-install
      rustup cargo-update
      python312Full micromamba pdm mypy ruff-lsp pipx
      nasm
      tailwindcss-language-server volta deno
      hvm kind2
      clang-tools lldb pkg-config
      # https://github.com/NixOS/nixpkgs/issues/331240
      niv nixpkgs-fmt nix-tree hub # fh
      marksman
      packwiz
      unityhub dotnet-sdk_8

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
        customCommands = [{
          key = "C";
          command = "git commit -m \"{{ .Form.Type }}{{if .Form.Scopes }}({{ .Form.Scopes }}){{end}}: {{ .Form.Description }}\" -s";
          description = "conventional commit";
          context = "files";
          prompts = [{
            type = "menu";
            title = "Select the type";
            key = "Type";
            options = [{
              name = "Feature";
              description = "a new feature";
              value = "feat";
            } {
              name = "Fix";
              description = "a bug fix";
              value = "fix";
            } {
              name = "Documentation";
              description = "Documentation only changes";
              value = "docs";
            } {
              name = "Styles";
              description = "Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)";
              value = "style";
            } {
              name = "Code Refactoring";
              description = "A code change that neither fixes a bug nor adds a feature";
              value = "refactor";
            } {
              name = "Performance Improvements";
              description = "A code change that improves performance";
              value = "perf";
            } {
              name = "Tests";
              description = "Adding missing tests or correcting existing tests";
              value = "test";
            } {
              name = "Builds";
              description = "Changes that affect the build system or external dependencies (example scopes: gulp, broccoli, npm)";
              value = "build";
            } {
              name = "Continuous Integration";
              description = "Changes that affect the build system or external dependencies (example scopes: gulp, broccoli, npm)";
              value = "ci";
            } {
              name = "Chores";
              description = "Other changes that don't modify src or test files";
              value = "chore";
            } {
              name = "Reverts";
              description = "Revert commit";
              value = "revert";
            }];
          } {
            type = "input";
            title = "Enter the scope(s) of this change.";
            key = "Scopes";
          } {
            type = "input";
            title = "Enter the short description of the change";
            key = "Description";
          } {
            type = "confirm";
            title = "Is the commit message correct?";
            body = "{{ .Form.Type }}{{if .Form.Scopes }}({{ .Form.Scopes }}){{end}}: {{ .Form.Description }}";
          }];
        }];
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
