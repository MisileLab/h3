#!/usr/bin/env bash
# Test runner script that sets up the correct environment for Nix

# Set up library path for zlib
export LD_LIBRARY_PATH=/nix/store/zph9xw0drmq3rl2ik5slg0n2frw9lw5m-zlib-1.3.1/lib:$LD_LIBRARY_PATH

# Set up Python path
export PYTHONPATH="/home/misile/repos/h3/projects/quantumdb/.venv/lib/python3.13/site-packages:/home/misile/repos/h3/projects/quantumdb"

# Run tests
exec /home/misile/repos/h3/projects/quantumdb/.venv/bin/pytest "$@"