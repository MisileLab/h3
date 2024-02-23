{ config, lib, pkgs, ... }:
{
  imports =
    [
    ];
  boot.loader.efi.canTouchEfiVariables = true;
}