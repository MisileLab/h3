#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import _exit

a, b = map(int, input().split(" "))

for i in range(2, a):
    if a % i == 0 and i < b:
        print(f"BAD {i}")
        _exit(0)
print("GOOD")

