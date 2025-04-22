#!/usr/bin/env bash
docker run --name pointer -v pointer:/etc/v2ray/config.json -p 10086:10086 v2fly/v2fly-core

