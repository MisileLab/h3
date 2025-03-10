#!/usr/bin/env bash
cd ./home-manager && rm result && nh home build -o result && attic push cache result && cd ..
cd ./system && rm result && nh os build -o result && attic push cache result && cd ..

