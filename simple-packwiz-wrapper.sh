#!/bin/bash

active=(modpacks/*)

if [ $1 = "export" ]
then
    cmd="packwiz mr export"
elif [ $1 = "refresh" ]
then
    cmd="packwiz refresh"
elif [ $1 = "update" ]
then
    cmd="packwiz update --all"
else
    cmd="packwiz mr add $2"
fi

for i in "${active[@]}"
do
    if [ -d "$i" ]; then
        echo "$i"
        cd "$i"
        $cmd
        cd -
    fi
done