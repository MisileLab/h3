#!/usr/bin/env bash

title=$(niri msg --json focused-window | jq -r 'if .title != null then .title | if length > 60 then .[0:57] + "..." else . end else "..." end')
printf '%s' "$title"
