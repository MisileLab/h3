# https://github.com/NixOS/nixpkgs/pull/305026
{ lib, stdenvNoCC, fetchurl, nodejs-slim, withNode ? false, testers }:

stdenvNoCC.mkDerivation (finalAttrs: {
  pname = "pnpm";
  version = "9.2.0";

  src = fetchurl {
    url = "https://registry.npmjs.org/pnpm/-/pnpm-${finalAttrs.version}.tgz";
    hash = "sha256-lPqyE98iHFW2lWsUoiZMIcYgPMqfCzuV/y/puEsSA5A=";
  };

  buildInputs = lib.optionals withNode [ nodejs-slim ];

  installPhase = ''
    runHook preInstall

    mkdir -p $out/{bin,libexec}
    cp -R . $out/libexec/pnpm
    ln -s $out/libexec/pnpm/bin/pnpm.cjs $out/bin/pnpm
    ln -s $out/libexec/pnpm/bin/pnpx.cjs $out/bin/pnpx

    runHook postInstall
  '';

  passthru = {
    tests.version = lib.optionalAttrs withNode (testers.testVersion {
      package = finalAttrs.finalPackage;
    });
  };

  meta = with lib; {
    description = "Fast, disk space efficient package manager";
    homepage = "https://pnpm.io/";
    license = licenses.mit;
    platforms = platforms.all;
    maintainers = [ ];
  };
})
