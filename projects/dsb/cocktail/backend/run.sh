#!/bin/bash
tor &
pdm run uvicorn main:app --port 80 &

wait -n
exit $?

