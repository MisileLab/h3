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

  home.sessionVariables = {
    QT_QPA_PLATFORM = "wayland";
    EDITOR = "nvim";
  };

  catppuccin.flavor = "mocha";

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
    fish = {
      enable = true;
      catppuccin.enable = true;
    };
    hyprland = {
      enable = true;
      catppuccin.enable = true;
      settings = {
        env = [
          "XDG_SESSION_TYPE,wayland"
          "QT_QPA_PLATFORM,wayland"
          "QT_QPA_PLATFORMTHEME,qt5ct"
          "XCURSOR_SIZE,24"
        ];
        exec-once = [
          "${pkgs.waybar}"
          "${pkgs.swww} init"
        ];
        animations = {
          enabled = "yes";
          bezier = [
            "myBezier,0.05,0.9,0.1,1.05"
            "overshot,0.05,0.9,0.1,0.1"
            "cubic,0.54,0.22,0.07,0.74"
            "overshotCircle,0.175,0.885,0.32,1.275"
            "bounce,1,1.6,0.1,0.85"
          ];
          animation = [
            "windowsIn,1,3,bounce,popin 23%"
            "windowsOut,1,7,bounce,slide"
            "windows,1,7,overshotCircle"
            "border,1,10,overshot"
            "borderangle,1,8,default"
            "fade,1,7,default"
            "workspaces,1,7,bounce"
          ];
          bindl = [
            ",XF86AudioMute,exec,bash ~/.scripts/volume mute"
            ",XF86AudioMicMute,exec,wpctl set-mute @DEFAULT_AUDIO_SOURCE@ toggle"
          ];
          bindle = [
            ",XF86AudioRaiseVolume, exec, bash ~/.scripts/volume up"
            ",XF86AudioLowerVolume, exec, bash ~/.scripts/volume down"
            ",XF86MonBrightnessUp,exec,~/.scripts/brightness up"
            ",XF86MonBrightnessDown,exec,~/.scripts/brightness down"
          ];
          bind = let mainMod = "SUPER"; in [
            "${mainMod}, RETURN, exec, kitty"
            "${mainMod}, C, killactive"
            "SUPERSHIFT, M, exit"
            "${mainMod}, V, togglefloating"
            "${mainMod}, D, exec, rofi -show drun"
            "${mainMod}, P, pseudo"
            "${mainMod}, J, togglesplit"
            "${mainMod}, left, movefocus, l"
            "${mainMod}, right, movefocus, r"
            "${mainMod}, up, movefocus, u"
            "${mainMod}, down, movefocus, d"
            "${mainMod}, 1, workspace, 1"
            "${mainMod}, 2, workspace, 2"
            "${mainMod}, 3, workspace, 3"
            "${mainMod}, 4, workspace, 4"
            "${mainMod}, 5, workspace, 5"
            "${mainMod}, 6, workspace, 6"
            "${mainMod}, 7, workspace, 7"
            "${mainMod}, 8, workspace, 8"
            "${mainMod}, 9, workspace, 9"
            "${mainMod}, 0, workspace, 10"
            "${mainMod} SHIFT, 1, movetoworkspace, 1"
            "${mainMod} SHIFT, 2, movetoworkspace, 2"
            "${mainMod} SHIFT, 3, movetoworkspace, 3"
            "${mainMod} SHIFT, 4, movetoworkspace, 4"
            "${mainMod} SHIFT, 5, movetoworkspace, 5"
            "${mainMod} SHIFT, 6, movetoworkspace, 6"
            "${mainMod} SHIFT, 7, movetoworkspace, 7"
            "${mainMod} SHIFT, 8, movetoworkspace, 8"
            "${mainMod} SHIFT, 9, movetoworkspace, 9"
            "${mainMod} SHIFT, 0, movetoworkspace, 10"
            "${mainMod}, mouse_down, workspace, e+1"
            "${mainMod}, mouse_up, workspace, e-1"
            "${mainMod}, mouse:272, movewindow"
            "${mainMod}, mouse:273, resizewindow"
            "SUPERSHIFT,S,exec,grim -g '$(slurp)' - | swappy -f -"
            "SUPER,F,fullscreen,1"
            "SUPERSHIFT,F,fullscreen,0"
            "SUPER,O,exec, waybar"
            "SUPER,R,exec, bash -c ~/.scripts/wall.sh"
            "SUPER,B,exec, bash ~/.scripts/bars.sh"
            "SUPER,X,exec, [float;resizeactive exact 450 800; centerwindow] kitty bash ~/.scripts/dmenu.sh"
          ];
          # I need to impl after https://github.com/Phant80m/Dotfiles/blob/main/.config/hypr/hyprland.conf effect section
        };
      };
    };
  };

  fonts.fontconfig.enable = true;

  # Let Home Manager install and manage itself.
  programs.home-manager.enable = true;
}
