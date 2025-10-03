{
  description = "QuantumDB - Neural compression-powered vector database";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, fenix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
pkgs = import nixpkgs {
  inherit system;
};

        rustToolchain = with fenix.packages.${system};
          combine (
            with stable;
            [
              clippy
              rustc
              cargo
              rustfmt
              rust-src
            ]
          );

        nativeBuildInputs = with pkgs; [
          rustToolchain
          pkg-config
          protobuf
          cargo-deny
          cargo-edit
          cargo-watch
          rust-analyzer
          cargo-criterion
        ];

        buildInputs = with pkgs; [
          openssl
          protobuf
        ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
          systemd
        ] ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
          darwin.apple_sdk.frameworks.Security
          libiconv
        ];

        quantumdb-package = pkgs.rustPlatform.buildRustPackage {
          pname = "quantumdb";
          version = "0.1.0";
          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          inherit nativeBuildInputs buildInputs;

          # Enable all features for the package
          cargoBuildFlags = [ "--all-features" ];

          # Run tests
          checkType = "release";

          # Python bindings
          pythonImportsCheck = [ "quantumdb" ];
        };

      in
      {
        # Development shell
devShells.default = pkgs.mkShell {
  buildInputs = buildInputs ++ (with pkgs; [ python3 python3Packages.numpy ]);
  inherit nativeBuildInputs;

  RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
  RUST_LOG = "debug";
  RUST_BACKTRACE = "1";
};

        # Package
        packages.default = quantumdb-package;

        # CLI tool
        packages.quantumdb-cli = quantumdb-package;

        # Python package
        packages.quantumdb-python = quantumdb-package;

        # Server
        packages.quantumdb-server = quantumdb-package;

        # Core library
        packages.quantumdb-core = quantumdb-package;

        # Apps
        apps.quantumdb = flake-utils.lib.mkApp {
          drv = quantumdb-package;
          exePath = "/bin/quantumdb";
        };

        # Overlay
        overlays.default = final: prev: {
          inherit rustToolchain;
        };
      });
}
