#!/usr/bin/env bash
rm ./home-manager/result
rm ./system/result
cd ./home-manager && nh home build . -o result && attic push cache result && cd ..
cd ./system && nh os build . -o result && attic push cache result && cd ..

