#!/bin/bash
mkdir /persist/tor
mkdir /persist/chats
ln -s /persist/tor /var/lib/tor
tor &
pdm run uvicorn main:app --port 80 &
ls /persist

wait -n
exit $?

