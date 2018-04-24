#!/usr/bin/env python3

import sys
from math import exp
from random import random, uniform, gauss
# import numpy as np


L = 2.0


def main():
    N = int(1e6)

    # uniform sampling
    S1 = 0.0
    for i in range(N):
        S1 += f(uniform(-L,L))
    S1 *= (L+L)/N

    # importance sampling with g(x)
    S2 = 0.0
    for i in range(N):
        while True:
            x = gauss(0.0, 1.0)
            # x = np.random.randn()
            if abs(x) <= L: break
        S2 += f(x)/g(x)
    S2 /= N

    # importance sampling with h(x)
    S3 = 0.0
    for i in range(N):
        while True:
            x = gauss(0.0, 0.6)
            # x = np.random.randn()*0.6
            if abs(x) <= L: break
        S3 += f(x)/h(x)
    S3 /= N

    print(S1, S2, S3)
    # NIntegrate in Wolfram|Alpha gives 1.6257386224450971730
    # could be an issue with the RNG

    return 0


def f(x):
    return exp(-0.5*x*x)/(1+x*x)


def g(x):
    # sigma = 1; sqrt(2*pi) ~= 2.5066282746310002
    # true normalization: \int _{-2} ^{+2} exp(-0.5*x*x) dx ~= 2.392576026645216
    return exp(-0.5*x*x)/2.392576026645216


def h(x):
    # sigma = 0.6; sqrt(2*pi*0.6) ~= 1.9416259125556992
    # true normalization: \int _{-2} ^{+2} exp(-x*x/1.2) dx ~= 1.9225527882257518
    return exp(-x*x/1.2)/1.9225527882257518


if __name__ == "__main__":
    sys.exit(main())
