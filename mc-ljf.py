#!/usr/bin/env python3

"""
classical Monte Carlo
for Lennard-Jones fluid
"""

import sys
import numpy as np
from math import exp, log1p, sqrt
from random import random, randrange


N = 64      # number of particles
L = 6.0     # supercell edge length

T = 2.0     # temperature

stepSize = 0.5      # adjust this for efficient flow through configuration space


def main():
    N_equil = int(1e3)
    N_steps = int(1e4)
    N_accpt = 0

    vol = L*L*L     # volume of the supercell
    rho = N/vol     # density
    beta = 1.0/T    # inverse temperature

    # np.random.seed(12345)           # for debugging, fix the internal rng state
    cnf = np.random.rand(N,3)*L     # initialize configuration

    # print(U)
    U = totalPotential(cnf)

    # equilibration block
    for t in range(N_equil):
        i = randrange(N)            # randomly choose an atom
        # trial displacement in PBC
        R_trial = np.mod(cnf[i] + np.random.randn(3)*stepSize, L)
        dU = changePotential(cnf, i, R_trial)       # calculate change in the potential
        # print(dU, beta*dU, exp(-beta*dU))
        # random() < exp(-beta*dU) ... causes overflow
        if -T*log1p(-random()) > dU:
            cnf[i,:] = R_trial[:]   # update the current cnf
            U += dU                 # update the potential

    # main block (data accumulation)
    U_data = list()
    for t in range(N_steps):
        i = randrange(N)            # randomly choose an atom
        # trial displacement in PBC
        R_trial = np.mod(cnf[i] + np.random.randn(3)*stepSize, L)
        dU = changePotential(cnf, i, R_trial)
        if -T*log1p(-random()) > dU:
            cnf[i,:] = R_trial[:]   # update the current cnf
            U += dU                 # update the potential
            N_accpt += 1            # count as accepted
        U_data.append(U)        # append potential to list

    print(N_accpt/N_steps, "acceptance")
    U_data = np.array(U_data)
    mu, serr, s, kappa = doStats(U_data)    # calculate statistics of time series
    print("E = %f +- %f" % (mu, serr))

    with open("final-cnf.xyz", "w") as f:
        f.write("%d\n\n" % (N))
        for i in range(N):
            f.write("H %f %f %f\n" % (cnf[i,0], cnf[i,1], cnf[i,2]))

    return 0


def changePotential(cnf, i, R_trial):
    dU = 0.0
    for j in range(i):
        r_mic = np.linalg.norm(np.mod(cnf[i] - cnf[j] + 0.5*L, L) - 0.5*L)
        rp_mic = np.linalg.norm(np.mod(R_trial - cnf[j] + 0.5*L, L) - 0.5*L)
        # minimum image convention (MIC)
        dU += LennardJones(rp_mic) - LennardJones(r_mic)
    for j in range(i+1,N):
        r_mic = np.linalg.norm(np.mod(cnf[i] - cnf[j] + 0.5*L, L) - 0.5*L)
        rp_mic = np.linalg.norm(np.mod(R_trial - cnf[j] + 0.5*L, L) - 0.5*L)
        # minimum image convention (MIC)
        dU += LennardJones(rp_mic) - LennardJones(r_mic)

    return dU


def totalPotential(cnf):
    U = 0.0
    for i in range(N):
        for j in range(i):
            r_mic = np.linalg.norm(np.mod(cnf[i] - cnf[j] + 0.5*L, L) - 0.5*L)
            # minimum image convention (MIC)
            U += LennardJones(r_mic)
    return U


# a horribly slow implementation of the Lennard-Jones potential
def LennardJones(r):
    return 4.0*(1.0/(r**12) - 1.0/(r**6))   # note reduced units


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
