#!/usr/bin/env python3

"""
variational Monte Carlo
for 3-dimensional harmonic oscillator
"""

import sys
import numpy as np
from math import exp, log1p, sqrt
# from random import random


alpha = 0.60
stepSize = 1.0      # adjust this for efficient flow through configuration space


def main():
    N_equil = int(1e3)
    N_steps = int(1e5)
    N_accpt = 0

    # np.random.seed(12345)           # for debugging, fix the internal rng state
    cnf = np.random.randn(3)*stepSize     # initialize configuration

    print("Initial coordinates:\n", cnf)
    U = totalMinusLnPsi2(cnf)

    # equilibration block
    for t in range(N_equil):
        # trial displacement
        # R_trial = cnf + np.random.randn(3)*stepSize
        R_trial = cnf + np.random.uniform(-0.5,0.5)*stepSize
        dU = changeMinusLnPsi2(cnf, R_trial)       # calculate change -ln Psi^2
        # print(dU, beta*dU, exp(-beta*dU))
        # if random() < exp(-dU):
        if -log1p(-np.random.random()) > dU:
            cnf[:] = R_trial[:]
            U += dU
        # print("e", U, dU)

    # main block (data accumulation)
    E_data = list()
    for t in range(N_steps):
        # trial displacement
        # R_trial = cnf + np.random.randn(3)*stepSize
        R_trial = cnf + np.random.uniform(-0.5,0.5)*stepSize
        dU = changeMinusLnPsi2(cnf, R_trial)       # calculate change -ln Psi^2
        if -log1p(-np.random.random()) > dU:
            cnf[:] = R_trial[:]
            U += dU
            N_accpt += 1
        E_data.append(localEnergy(cnf))    # append energy to list
        # print("m", U, dU)

    print(N_accpt/N_steps, "acceptance")
    E_data = np.array(E_data)
    mu, serr, s, kappa = doStats(E_data)    # calculate statistics of time series
    print("E = %f +- %f" % (mu, serr))

    print("Final coordinates:\n", cnf)

    return 0


def changeMinusLnPsi2(cnf, R_trial):
    r_ = np.linalg.norm(R_trial)
    r = np.linalg.norm(cnf)
    return 2.0*alpha*(r_*r_ - r*r)


def totalMinusLnPsi2(cnf):
    r = np.linalg.norm(cnf)
    return 2.0*alpha*r*r


def localEnergy(cnf):
    r = np.linalg.norm(cnf)
    return 3.0*alpha + r*r*(0.5 - 2.0*alpha*alpha)


def doStats(a):
    mu = np.mean(a)
    s = np.std(a)
    ac = np.correlate(a-mu, a-mu, mode="full")
    ac /= ac[int(len(ac)>>1)]   # autocorrelation function
    kappa = 1.0
    for j in range(int(len(ac)>>1)+1, int(len(ac))):
        if ac[j] < 0:
            break
        else:
            kappa += (2.0*ac[j])
            # autocorrelation time
    serr = s*(kappa/len(a))**(0.5)

    return mu, serr, s, kappa


if __name__ == "__main__":
    sys.exit(main())
