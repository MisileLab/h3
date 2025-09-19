{pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";
}
