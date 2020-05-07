#!/usr/bin/env python3

"""
kinetic Monte Carlo
for crystal growth
"""

import sys
import numpy as np


Lx = 8
Ly = 8
lat2d = np.zeros((Lx, Ly), dtype=np.uint)
LL = Lx * Ly

# assume kB*T = 1 ... ie. "absorb"
phi_ss = 0.15
phi_ff = phi_sf = 0.0
# delta_mu = 2.0
gamma = 2*phi_ff+2*phi_ss-4*phi_sf     # positive if growth is favoured
xi = np.exp(0.5*gamma)

p_cnt = 3       # deposit, evaporate, diffuse
u_thr = np.array([1.0, 1.0])
u_thr /= (u_thr.sum() + np.exp(-gamma))
# print(u_thr)


def main():
    N_equil = int(1e3)
    # N_steps = int(1e4)
    # N_accpt = 0

    # np.random.seed(12345)   # for debugging, fix the internal rng state
    global lat2d
    # lat2d += 10000
    buf = [0,0,0]

    for t in range(N_equil):
        r = divmod(np.random.randint(Lx*Ly), Ly)
        runproc_hardwired(r)

    print(lat2d)
    return 0


def runproc_hardwired(r):
    u = np.random.random()

    # TODO: adjust process weights
    r_e = (r[0]+1   )%Lx,  r[1]
    r_w = (r[0]-1+Lx)%Lx,  r[1]
    r_n =  r[0]         , (r[1]+1   )%Ly
    r_s =  r[0]         , (r[1]-1+Ly)%Ly

    nbrs = get_neighbours(r)
    n_nbr = len(nbrs)
    # print(r, n_nbr)

    if u < u_thr[0]:    # deposit
        v = np.random.random()
        if v < xi**(n_nbr-2):
            lat2d[r] += 1
    elif u < u_thr[1]:  # evaporate
        if lat2d[r] == 0:   # implies infinite binding energy for substrate
            return

        v = np.random.random()
        if v < xi**(2-n_nbr):
            lat2d[r] -= 1
    else:               # diffuse
        if lat2d[r] == 0:   # implies infinite binding energy for substrate
            return

        lat2d[r] -= 1
        v = np.random.randint(4)
        if v == 0:
            lat2d[r_e] += 1
        elif v == 1:
            lat2d[r_w] += 1
        elif v == 2:
            lat2d[r_n] += 1
        else:
            lat2d[r_s] += 1


def get_neighbours(r):
    r_e = (r[0]+1   )%Lx,  r[1]
    r_w = (r[0]-1+Lx)%Lx,  r[1]
    r_n =  r[0]         , (r[1]+1   )%Ly
    r_s =  r[0]         , (r[1]-1+Ly)%Ly

    nbrs = list()
    if lat2d[r] <= lat2d[r_e]:
        nbrs.append(0)
    if lat2d[r] <= lat2d[r_w]:
        nbrs.append(1)
    if lat2d[r] <= lat2d[r_n]:
        nbrs.append(2)
    if lat2d[r] <= lat2d[r_s]:
        nbrs.append(3)

    return nbrs


# def pickproc(r):    # generalized
#     u = np.random.random()
#     # TODO: adjust process weights
# 
#     for p in range(p_cnt-1):
#         if u < u_thr[p]:
#             return p
#     return p_cnt-1


if __name__ == '__main__':
    sys.exit(main())
