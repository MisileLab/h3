{
  description = "Flutter 3.13.x";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            android_sdk.accept_license = true;
            allowUnfree = true;
          };
        };
        buildToolsVersion = "35.0.0";
        androidComposition = pkgs.androidenv.composeAndroidPackages {
          buildToolsVersions = [ buildToolsVersion "29.0.3" ];
          platformVersions = [ "35" "29" ];
          abiVersions = [ "armeabi-v7a" "arm64-v8a" ];
        };
        androidSdk = androidComposition.androidsdk;
      in
      {
        devShell =
          with pkgs; mkShell {
            ANDROID_SDK_ROOT = "${androidSdk}/libexec/android-sdk";
            buildInputs = [
              flutter
              androidSdk # The customized SDK that we've made above
              jdk17
              pkg-config
              # C++ development dependencies
              clang
              llvmPackages.libcxx
              llvmPackages.libunwind
              gtk3
              gtk3.dev
              glib
              glib.dev
              sysprof
              # Alternative: use stdenv.cc instead of clang
              # stdenv.cc
              # stdenv.cc.cc.lib
            ];
            # Set up C++ environment
            shellHook = ''
              export CC=clang
              export CXX=clang++
            '';
          };
      });
}
