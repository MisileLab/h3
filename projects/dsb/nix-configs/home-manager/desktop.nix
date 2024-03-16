{ config, pkgs, catppuccin, ... }:
let 
  c = import ./config.nix;
  custom-ctps = {
    waybar = builtins.fetchGit{url="https://github.com/catppuccin/waybar.git";rev="f74ab1eecf2dcaf22569b396eed53b2b2fbe8aff";};
    dunst = builtins.fetchGit{url="https://github.com/catppuccin/dunst.git";rev="a72991e56338289a9fce941b5df9f0509d2cba09";};
    delta = builtins.fetchGit{url="https://github.com/catppuccin/delta";rev="21b37ac3138268d92cee71dfc8539d134817580a";};
  };
  returnColorCSS = {r, g, b, a, addi ? ""}: ''
    ${(if c.gtk4 then "backdrop-filter: blur(5px)" else "")}
    background: rgba(${toString r}, ${toString g}, ${toString b}, ${toString a});
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    border-radius: 0;
    ${addi}
  '';
in
{
  home.pointerCursor = {
    name = "Adwaita";
    package = pkgs.gnome.adwaita-icon-theme;
    size = 32;
  };

  i18n = {
    inputMethod = {
      enabled = "kime";
      kime.config = {
        indicator.icon_color = "White";
      };
    };
  };

  wayland.windowManager.sway = {
    enable = true;
    extraConfigEarly = ''
      blur enable
      blur_xray enable
      blur_radius 10
      corner_radius 8
      blur_passes 4
      shadows enable
      for_window [class=\".*\"] opacity 0.9
      for_window [app_id=\".*\"] opacity 0.9
      bindsym Print exec ${pkgs.sway-contrib.grimshot}/bin/grimshot --notify copy area
      bindsym Shift+Print	exec ${pkgs.sway-contrib.grimshot}/bin/grimshot --notify copy screen
      bindsym Shift+Alt+Print exec ${pkgs.sway-contrib.grimshot}/bin/grimshot --notify savecopy anything

      bindsym Alt+Print exec ${pkgs.sway-contrib.grimshot}/bin/grimshot --notify save area
      
      bindsym XF86AudioRaiseVolume exec ${pkgs.avizo}/bin/volumectl -u up
      bindsym XF86AudioLowerVolume exec ${pkgs.avizo}/bin/volumectl -u down
      bindsym XF86AudioMute exec ${pkgs.avizo}/bin/volumectl toggle-mute
      bindsym XF86AudioMicMute exec ${pkgs.avizo}/bin/volumectl -m toggle-mute

      bindsym XF86MonBrightnessUp exec ${pkgs.avizo}/bin/lightctl up
      bindsym XF86MonBrightnessDown exec ${pkgs.avizo}/bin/lightctl down

      bindsym Mod1+y exec ${pkgs.clipman}/bin/clipman pick --tool="rofi" --max-items=30

      exec "${pkgs.avizo}/bin/avizo-service"
    '';
    config = {
      bars = [];
      startup = [
        {command = "${pkgs.waybar}/bin/waybar";}
        {command = "${pkgs.swaybg}/bin/swaybg --image ~/.config/home-manager/bg.png";}
        {command = "${pkgs.wl-clipboard}/bin/wl-paste -t text --watch ${pkgs.clipman}/bin/clipman store --no-persist";}
      ];
      menu = "${pkgs.rofi-wayland}/bin/rofi -show drun";
      terminal = "${pkgs.alacritty}/bin/alacritty";
      fonts = {
        names = ["Fira Code NF" "Fira Code" "NanumSquare"];
        style = "Regular";
        size = 12.0;
      };
      window.titlebar = false;
      modifier = "Mod4";
    };
    package = pkgs.swayfx;
    xwayland = true;
  };

  programs = {
    swaylock = {
      enable = true;
      settings = {
        # https://github.com/catppuccin/swaylock
        color = "1e1e2e";
        bs-hl-color = "f5e0dc";
        caps-lock-bs-hl-color = "f5e0dc";
        caps-lock-key-hl-color = "a6e3a1";
        inside-color = "00000000";
        inside-clear-color = "00000000";
        inside-caps-lock-color = "00000000";
        inside-ver-color = "00000000";
        inside-wrong-color = "00000000";
        key-hl-color = "a6e3a1";
        layout-bg-color = "00000000";
        layout-border-color = "00000000";
        layout-text-color = "cdd6f4";
        line-color = "00000000";
        line-clear-color = "00000000";
        line-caps-lock-color = "00000000";
        line-ver-color = "00000000";
        line-wrong-color = "00000000";
        ring-color = "b4befe";
        ring-clear-color = "f5e0dc";
        ring-caps-lock-color = "fab387";
        ring-ver-color = "89b4fa";
        ring-wrong-color = "eba0ac";
        separator-color = "00000000";
        text-color = "cdd6f4";
        text-clear-color = "f5e0dc";
        text-caps-lock-color = "fab387";
        text-ver-color = "89b4fa";
        text-wrong-color = "eba0ac";
      };
    };
    zellij = {
      enable = true;
      enableFishIntegration = true;
      settings = {
        theme = "catppuccin-mocha";
        default_shell = "fish";
      };
    };
    git = {
      delta.options = {
        features = "catpuccin-mocha";
      };
      includes = [{path = "${custom-ctps.delta}/themes/mocha.gitconfig";}];
    };
    helix = {
      enable = true;
      catppuccin.enable = true;
      languages = {
        language = [{
          name = "python";
          auto-format = false;
          indent = {tab-width = 2; unit = " ";};
        } {
          name = "jsx";
          language-servers = ["tailwindcss-ls" "typescript-language-server"];
        } {
          name = "tsx";
          language-servers = ["tailwindcss-ls" "typescript-language-server"];
        } {
          name = "svelte";
          language-servers = ["tailwindcss-ls" "svelteserver"];
        }];
      };
    };
    fish = {
      catppuccin.enable = true;
      plugins = [{name="tide"; src=pkgs.fishPlugins.tide.src;}];
    };
    alacritty = {
      enable = true;
      catppuccin.enable = true;
      settings = {shell = "${pkgs.fish}/bin/fish";};
    };
    waybar = {
      enable = true;
      settings = [{
        modules-left = [ "sway/workspaces" ];
        modules-center = [ "sway/window" ];
        modules-right = [ "temperature" "pulseaudio" "network" "battery" "cpu" "memory" "clock"];
        backlight = {
          device = "intel_backlight";
          format = "{icon} {percent}%";
          format-icons = ["" ""];
        };
        cpu = {
          interval = 5;
          format = " {}%";
          max-length = 10;
        };
        memory = {
          interval = 5;
          format = " {}%";
          tooltip-format = "{used}G/{total}G";
          max-length = 10;
        };
        network = {
          interface = "wlan0";
          format = "{ifname}";
          format-wifi = "  {signalStrength}%";
          format-ethernet = "󰊗  {ipaddr}";
          format-disconnected = "Disconnected";
          tooltip-format = "{ifname} via {gwaddr} 󰊗";
          tooltip-format-wifi = "{essid} ({signalStrength}%) ";
          tooltip-format-ethernet = "{ifname} ";
          tooltip-format-disconnected = "Disconnected";
          max-length = 40;
      };
      clock = {
        interval = 1;
        format = "{:%H:%M:%S}";
        tooltip-format = "{:%Y-%m-%dT%H:%M:%S}";
        max-length = 25;
      };
      battery = {
        bat = "BAT1";
        interval = 60;
        states = {
          warning = 30;
          critical = 15;
        };
        format = "{icon} {capacity}%";
        format-icons = [" " " " " " " " " "];
        max-length = 25;
      };
      "sway/window" = {
        format = "{app_id} - {title}";
        max-length = 50;
      };
    }];
    style = ''
      @import "${custom-ctps.waybar}/themes/mocha.css";
      * {
        font-family: 'Fira Code', monospace;
      }
      window#waybar {${returnColorCSS({r=108;g=112;b=134;a=0.4;})}}
      #workspaces button {
        ${returnColorCSS({r=108;g=112;b=134;a=0.4;addi=''
        '';})}
      }
      #workspaces button.active {
        ${returnColorCSS({r=147;g=153;b=178;a=0.6;})}
      }
      #workspace button.urgent {
        background: rgba(249, 226, 175, 0.6);
      }
      #workspace button.hover {
        ${returnColorCSS({r=127;g=132;b=156;a=0.6;addi=''
        '';})}
      }
      .modules-left * {
        color: @subtext1;
        padding-left: 0px;
        padding-right: 0px;
      }
      .modules-center * {
        color: @text;
      }
      .modules-right * {
        ${returnColorCSS({r=147;g=153;b=178;a=0.6;addi=''
          padding-left: 6px;
          padding-right: 6px;
          color: @subtext1;
        '';})}
      }
      #battery.warning {
        color: @yellow;
      }
      #battery.critical {
        color: @red;
      }'';
    };
    firefox.enable = true;
    rofi = {
      enable = true;
      extraConfig = {
        modi = "run,drun,window";
        icon-theme = "Oranchelo";
        show-icons = true;
        terminal = "alacritty";
        drun-display-format = "{icon} {name}";
        location = 0;
        hide-scrollbar = true;
        display-drun = "   Apps ";
        display-run = "   Run ";
        display-window = " 﩯  Window";
        disable-history = false;
        display-Network = " 󰤨  Network";
        sidebar-mode = true;
      };
      theme = "catppuccin-mocha";
      package = pkgs.rofi-wayland;
    };
  };
  services = {
    dunst = {
      enable = true;
      configFile = "${custom-ctps.dunst}/src/mocha.conf";
    };
    avizo.enable = true;
    poweralertd.enable = true;
    gpg-agent = {
      enable = true;
      enableSshSupport = true;
      extraConfig = "pinentry-program ${pkgs.pinentry.curses}/bin/pinentry-curses";
    };
  };
}
