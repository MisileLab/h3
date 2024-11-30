with import <nixpkgs> {};
mkShell {
  NIX_LD_LIBRARY_PATH = lib.makeLibraryPath [
    stdenv.cc.cc
    stdenv.cc.cc.lib
    openssl
    libseccomp
    libunwind
  ];
  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
  NIX_LD = lib.fileContents "${stdenv.cc}/nix-support/dynamic-linker";
}
