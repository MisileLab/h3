{pkgs, ...}: {
  home.packages = with pkgs; [
    # file
    axel file wgetpaste ouch duperemove fd

    # some fancy cli tools
    nix-output-monitor nix-forecast

    # process utils
    killall
  ];
  programs = {
    fastfetch.enable = true;
    ripgrep.enable = true;
    eza.enable = true;
    bat.enable = true;
    btop.enable = true;
  };
}
