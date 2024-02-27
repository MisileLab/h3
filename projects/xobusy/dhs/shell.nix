with import <nixpkgs> {};
  mkShell {
    nativeBuildInputs = with pkgs.buildPackages; [libseccomp] ++
    (with python311Packages; [numpy httpx python-lsp-server pwntools beautifulsoup4 lxml loguru]);
    NIX_LD_LIBRARY_PATH = lib.makeLibraryPath [
      stdenv.cc.cc
      libseccomp
    ];
    NIX_LD = lib.fileContents "${stdenv.cc}/nix-support/dynamic-linker";
  }
