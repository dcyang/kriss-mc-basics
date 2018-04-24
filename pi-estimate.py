#!/usr/bin/env python3

from random import random

N = int(1e8)
k = 0

for i in range(N):
    x, y = random(), random()
    if x*x + y*y < 1.0:
        k += 1

print("%.8f" % (4*k/N))
