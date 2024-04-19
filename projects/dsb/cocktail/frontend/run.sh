#!/bin/bash
tor &
pnpm run dev --port 80 &
python print-tor.py &

sleep infinity
