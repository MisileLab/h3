{ pkgs ? import <nixpkgs> {} }:

(pkgs.buildFHSEnv {
  name = "simple-x11-env";
  targetPkgs = pkgs: (with pkgs; [
    libgcc
  ]); /*(with pkgs.xorg; [
    libX11
    libXcursor
    libXrand
  ]);
  multiPkgs = pkgs: (with pkgs; [
  ]);*/
  runScript = "fish";
}).env
