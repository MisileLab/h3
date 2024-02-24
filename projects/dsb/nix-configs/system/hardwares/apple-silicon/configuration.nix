{ config, lib, pkgs, ... }:
{
  imports =
    [
      ../../common.nix
      <apple-silicon-support/apple-silicon-support>
      ./hardware-configuration.nix
    ];
  boot.loader.efi.canTouchEfiVariables = false;
  hardware.asahi = {
    withRust = true;
    useExperimentalGPUDriver = true;
    addEdgeKernelConfig = true;
    experimentalGPUInstallMode = "replace";
  };

  services.actkbd={enable=true;bindings=[{keys=[225];events=["key"];command="/run/current-system/sw/bin/light -A 10";} {keys=[224];events=["key"];command="/run/current-system/sw/bin/light -U 10";}];};
  programs.light.enable = true;
}

