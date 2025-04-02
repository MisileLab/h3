#!/usr/bin/env sh

sed "s/PLACEHOLDERFORJSON/$(uuidgen)" /etc/v2ray/config.json
/usr/bin/v2ray

