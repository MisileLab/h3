{pkgs, ...}: {
  home.packages = with pkgs; [
    # file
    axel wget file wgetpaste convmv termscp ouch rhash duperemove

    # some fancy cli tools
    fastfetch onefetch delta genact glow dasel navi

    # process utils
    killall screen
  ];
  programs = {
    mpv = {
      enable = true;
      catppuccin.enable = true;
    };
    zoxide = {
      enable = true;
      options = ["--cmd cd"];
    };
    fzf.enable = true;
    tealdeer.enable = true;
    ripgrep.enable = true;
    eza.enable = true;
    bat = {
      enable = true;
      catppuccin.enable = true;
    };
    atuin = {
      enable = true;
      enableFishIntegration = true;
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
        default_shell = "fish";
      };
    };
    git.delta = {
      enable = true;
      catppuccin.enable = true;
    };
  };
}
