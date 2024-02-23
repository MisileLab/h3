{ config, lib, pkgs, ... }:
{
  imports =
    [
      <apple-silicon-support/apple-silicon-support>
    ];
  boot.loader.efi.canTouchEfiVariables = false;
  hardware.asahi.useExperimentalGPUDriver = true;
  hardware.asahi.withRust = true;

  services.actkbd={enable=true;bindings=[{keys=[225];events=["key"];command="/run/current-system/sw/bin/light -A 10";} {keys=[224];events=["key"];command="/run/current-system/sw/bin/light -U 10";}];};
  programs.light.enable = true;
}

