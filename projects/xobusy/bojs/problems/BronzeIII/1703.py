#!/usr/bin/env python
# -*- coding: utf-8 -*-

while True:
  list_ = list(map(int, input().split(" ")))
  leaf = 1
  if list_[0] == 0:
    break
  
  for i in range(1, len(list_), 2):
    leaf = leaf * list_[i] - list_[i+1]
  print(leaf)

