{pkgs, ...}: {
  home.packages = with pkgs; [
    # file
    axel wget file wgetpaste convmv termscp ouch rhash duperemove
    fd cloc

    # some fancy cli tools
    fastfetch onefetch delta genact glow navi nix-output-monitor
    nix-forecast

    # process utils
    killall screen
  ];
  programs = {
    mpv.enable = true;
    zoxide = {
      enable = true;
      options = ["--cmd cd"];
    };
    fzf.enable = true;
    tealdeer.enable = true;
    ripgrep.enable = true;
    eza = {
      enable = true;
      enableNushellIntegration = false;
    };
    bat.enable = true;
    atuin = {
      enable = true;
      enableNushellIntegration = true;
      settings = {
        history_filter = [
          "ls" "clear" "cd *" "manual" "poweroff" "zellij *" "exit" "topgrade" "zellij *" "lzg"
          "navi*" "mullvad*"
        ];
      };
    };
    zellij = {
      enable = true;
      settings = {
        theme = "catppuccin-mocha";
        default_shell = "nu";
      };
    };
    git.delta.enable = true;
  };
}
