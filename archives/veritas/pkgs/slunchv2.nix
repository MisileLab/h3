{
  stdenv,
  lib,
  fetchFromGitHub,
  bun,
  makeBinaryWrapper,
  autoPatchelfHook,
  libgcc,
}:
let
  version = "0.1.0";
  src = fetchFromGitHub {
    owner = "ny0510";
    repo = "slunchv2-backend";
    rev = "v${version}";
    hash = "sha256-kmkSaRBwiWLATLPRFTKo6WdYM2RhJJWPrS6yFNeJXlQ=";
  };
  node_modules = stdenv.mkDerivation {
    pname = "slunchv2-node_modules";
    inherit src;
    version = version;
    impureEnvVars = lib.fetchers.proxyImpureEnvVars;
    nativeBuildInputs = [ bun autoPatchelfHook stdenv.cc.cc ];
    dontConfigure = true;
    dontFixup = true;
    dontPatchShebangs = true;
    buildPhase = ''
      bun install --no-progress --frozen-lockfile --production --ignore-scripts

      autoPatchelf /build/source/node_modules/.bin/node-gyp-build-optional-packages

      bun install --production
    '';
    installPhase = ''
      mkdir -p $out/node_modules

      cp -R ./node_modules $out
    '';
    outputHash = "sha256-6Lia8m3GM3QE4XBb5v9YM97KIx9aPFNd8Apn/cTRgRM=";
    outputHashAlgo = "sha256";
    outputHashMode = "recursive";
  };
in
stdenv.mkDerivation {
  pname = "slunchv2";
  version = version;
  inherit src;
  nativeBuildInputs = [ makeBinaryWrapper ];

  dontConfigure = true;
  dontBuild = true;

  installPhase = ''
    runHook preInstall

    mkdir -p $out/bin

    ln -s ${node_modules}/node_modules $out
    cp -R ./* $out

    # bun is referenced naked in the package.json generated script
    makeBinaryWrapper ${bun}/bin/bun $out/bin/slunchv2 \
      --prefix PATH : ${lib.makeBinPath [ bun ]} \
      --prefix LD_LIBRARY_PATH : ${libgcc.lib}/lib \
      --add-flags "run --prefer-offline --no-install --cwd $out ./src/index.ts"

    runHook postInstall
  '';

  meta = with lib; {
    homepage = "https://github.com/ny0510/slunchv2-backend";
    changelog = "https://github.com/ny0510/slunchv2-backend/releases/tag/${src.rev}";
    description = "Backend of slunchv2";
    mainProgram = "slunchv2";
    maintainers = with maintainers; [ misilelab ];
    license = with licenses; [ gpl3Plus ];
    platforms = [
      "x86_64-linux"
    ];
  };
}
