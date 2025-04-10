#!/usr/bin/env sh
nom build -L .#nixosConfigurations.veritas.config.system.build.toplevel --show-trace --option binary-caches https://cache.nixos.org

