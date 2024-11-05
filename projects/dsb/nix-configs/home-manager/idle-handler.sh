#!/usr/bin/env bash

# Check the current idle_inhibit status
STATUS=$(swaymsg -t get_tree | jq '.. | select(.type?) | select(.inhibit_idle?) | .inhibit_idle')

if [ "$STATUS" == "" ]; then
    # Enable idle inhibit
    swaymsg "inhibit_idle visible"
    notify-send "Idle Inhibit Enabled"
else
    # Disable idle inhibit
    swaymsg "inhibit_idle none"
    notify-send "Idle Inhibit Disabled"
fi

