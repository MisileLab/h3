#!/bin/bash
mkdir /persist/tor
mkdir /persist/chats
ln -s /persist/tor /var/lib/tor
tor &
python print-tor.py &
pdm run uvicorn main:app --port 80 &

sleep infinity

