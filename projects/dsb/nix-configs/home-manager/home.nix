{ config, pkgs, ... }:
{
  home.username = "misile";
  home.homeDirectory = "/home/misile";
  home.stateVersion = "23.11"; # dont change it

  home.packages = with pkgs; [
    # Network
    dhcpcd

    # Fonts
    fira-code-nerdfont nanum
  ];

  home.file = {};

  home.sessionVariables = {};

  services = {
    dunst = {
      enable = true;
      settings = {
        global = {
          font = "FiraCode Nerd Font Mono";
          allow_markup = "yes";
          format = "<b>%s</b>\n%b";
          sort = "yes";
          indicate_hidden = "yes";
          alignment = "left";
          bounce_freq = 0;
          ellipsize = "middle";
          show_age_thresold = 10;
          word_wrap = "yes";
          Ignore_newline = "no";
          width = 320; height = 100;
          origin = "top-right";
          offset = "21x21";
          progress_bar = true;
          prograss_bar_height = 10;
          progress_bar_frame_width = 0;
          progress_bar_min_width = 150;
          progress_bar_max_width = 300;
          frame_width = 2;
          frame_color = "#89B4FA";
          transparency = 30;
          idle_thresold = 0;
          follow = "mouse";
          show_indicators = "no";
          sticky_history = "yes";
          line_height = 8;
          separator_height = 3;
          separator_color = "frame";
          padding = 10;
          horizontal_padding = 12;
          text_icon_padding = 16;
          startup_notification = false;
          icon_position = "left";
          min_icon_size = 32;
          max_icon_size = 48;
          corner_radius = 13;
          always_run_script = true;
          mouse_left_click = "close_current";
        	mouse_middle_click = "do_action, close_current";
        	mouse_right_click = "close_all";
        };
        urgency_low = {
          background = "#1E1E2E";
          foreground = "#CDD6F4";
        };
        urgency_normal = {
          background = "#1E1E2E";
          foreground = "#CDD6F4";
        };
        urgency_critical = {
          background = "#1E1E2E";
          foreground = "#CDD6F4";
          frame_color = "#FAB387";
        };
      };
    };
  };

  programs = {
    atuin = {
      enable = true;
      settings = {
        style = "full";
        inline_height = 20;
        auto_sync = true;
        sync_frequency = "1h";
      };
    };
  };

  fonts.fontconfig.enable = true;

  # Let Home Manager install and manage itself.
  programs.home-manager.enable = true;
}
