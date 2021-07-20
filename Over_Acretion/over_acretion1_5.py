""" This is my first program for solving heat equation in rocks """

# 1.5 % h2o

# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
from scipy.linalg import solve_banded
from numpy.linalg import inv
import pylab as pl
import matplotlib.pyplot as plt
import copy

Nx = 641  # Number of grid points in X including boundary points

Nt = 3203  # Number of timesteps to compute

L = 16000  # size of initial spacial domain in meters

dx = float((L / (Nx - 1.0)))  # Calculate Spatial Step-Size, grid-Size.
print ("este es dx", dx)

# Initial layers size

lr = float(20000.0)  # Upper crust size in (m) meters.
lm = float(50.0)  # Fist magma emplacement size in m.
la = float(20000.0)  # Mantle size in meters
li = float(50.0)  # subsequent magma injections size in m

Lr = int((lr / dx) - 1)  # Lr is the number of equations for the rock domain without interception point.
Lm = int((lm / dx))  # Lm is the number of equations for the magma domain.
LI = int((li / dx))  # Li is the number of equations for the new injections domain.
La = int((la / dx))  # number of grid point for mantle
n = int(Lr + Lm + La + 1)  # total size of the matrix, and is equal to Nx
t = float((1.009152 * 10 ** (14)))  # Total time for thermal evolution.
dt = float(t / Nt)  # Create Temporal Step-Size, TFinal, Number of Time-Steps

# Initial Conditions

gt = float(400.0 / (Lr + 1))  # Geothermal gradient
Lr = int(Lr)
b = np.zeros((n, 1))

b[0, 0] = gt
for i in range(1, Lr, 1):
    b[i, 0] = float(b[i - 1, 0] + gt)

b[Lr, 0] = float(0.0)

for i in range(Lr + 1, Lr + Lm, 1):  # first magma injection temperature.
    b[i, 0] = 1302.0

for i in range(Lr + Lm + 1, n, 1):
    b[Lr + Lm, 0] = float(400.0)
    b[i, 0] = float(b[i - 1, 0] + gt)

# new may 9th. Create constants for lower crust

kl = float(2.6)
cl = float(1200.0)
dl = float(2900)
LL = float(2.93 * 10 ** 5)
TLc= float(1200.0)
Tsc= float(822.0)
Tac= float(981.5)


# Create constants for equations of upper crust .

kr = float(3.0)  # Thermal conductivity of the upper crust J/mKS
Cc = float(1370.0)  # Specific heat capacity J/Kg
dc = float(2650.0)  # upper crust density Kg/m3
TLR = float(1150.0)  # Rock Liquidus temperature in C
TsR = float(812.0)  # Rock solidus temperature in C
TaR = float(900.0)
LR = float(2.7 * 10 ** 5)  # Latent heat from upper crust and mantle.

# Finite differences equations constants for upper crust, quite kr para poner pr

Sr = float((1 / (dx * dx)))

dr = float(((dc * Cc) / dt) + ((dc * LR * 6.36 * 10 ** (-3)) / dt))  # Constant for TLR and 900

dr1 = float(((dc * Cc) / dt) + ((dc * LR * 1.75 * 10 ** (-3)) / dt))  # Constant for T and TsR

S = float((1 * dt) / (dc * Cc * dx * dx))  # S de la ecuacion de la roca

# Finite differences equations constants for lower crust

Srl = float((kl / (dx * dx)))

drl = float(((dl * cl) / dt) + ((dl * LL * 3.2392 * 10 ** (-3)) / dt))  # Constant for TLR and 981

dr1l = float(((dl * cl) / dt) + ((dl * LL * 1.832* 10 ** (-3)) / dt))  # Constant for T and TsR

Sl = float((kl * dt) / (dl * cl * dx * dx))  # S de la ecuacion de la roca

# Create constants for magma matrix.


km = 2.6  # Thermal conductivity of the magma J/mKS
Rom = 2830.0  # Injected basalt density Kg/m3
Cpm = 1480.0  # Specific heat capacity J/Kg
L = float(4.0 * 10 ** 5)  # latent heat of injected magma
Ta = float(1075.0)
TL = float(1302.0)  # Basalt  Liquidus temperature in C
Ts = float(720.0)  # Basalt solidus temperature in C

# constat for heat eq above liquidus below solidus ( for a while)
S1 = float((km * dt) / (Rom * Cpm * dx * dx))

# Constants for eq between Ta and Tl

S2 = float((km / (dx * dx)))
d = float(((Rom * Cpm) / dt) + ((Rom * L * 3.25 * 10 ** (-3)) / dt))

# Constants for eq between Ta and Ts S2 is still a constant

Xa = float(((3.25 * 10 ** (-3)) * (Ta - TL)) + 1)
m = float(Xa / (Ta - Ts))
d1 = float(((Rom * Cpm) / dt) + ((Rom * L * m) / dt))

# Dirichlet Boundary condition

C = np.zeros((n, 1))
C[0, 0] = float(S * 1)

a = np.zeros((n, n))  # Create matrix to be solved (empty)

# Open memory space for data.
data = []
data2 = []
data3 = []
data4 = []
data5 = []
data6 = []
data7 = []

crust1 = []

Tcr1 = []

Ti1 = []

X1 = []
X2 = []
X3 = []
X4 = []
X5 = []
X6 = []
X7 = []

F1 = []
F2 = []
F3 = []
F4 = []
F5 = []
F6 = []
F7 = []

b = b + C  # sumar la condicion de dirichlet

# Evaluate the matrix in the time.
dT = 0
dT1 = 0
dT2 = 0
dT3 = 0
dT4 = 0
dT5 = 0

dt1 = 0

DT = int((3.1536 * 10 ** 11) / dt)  # Number of Time steps for the first magma injection.
u = 0
r = float(40050.0)

for j in range(1, Nt, 1):  # loop for solving system

    # heat transfer  equations for  Upper crust  with phase change possibility.
    # The matrix a is the system of linear equations.

    for c in range(0, Lr, 1):
        if b[c, 0] > TLR:  # if rock temperature is greatest than T liquidus
            pr = float(kr * (1 + (1.5 * c * 25 * 10 ** (-6))) / (1 + (1.5 * b[c, 0] * 10 ** (-3))))
            a[c, c] = float(1 + (2 * S* pr))
            a[c, c + 1] = -S*pr
            if c > 0:
                pr = float(kr * (1 + (1.5 * c * 25 * 10 ** (-6))) / (1 + (1.5 * b[c, 0] * 10 ** (-3))))
                a[c, c - 1] = -S * pr
        elif b[c, 0] <= TLR and b[c, 0] >= TaR:
            pr = float(kr * (1 + (1.5 * c * 25 * 10 ** (-6))) / (1 + (1.5 * b[c, 0] * 10 ** (-3))))
            a[c, c] = float(((2 * Sr * pr) + dr) / dr)
            a[c, c + 1] = (- Sr * pr) / dr
            if c > 0:
                pr = float(kr * (1 + (1.5 * c * 25 * 10 ** (-6))) / (1 + (1.5 * b[c, 0] * 10 ** (-3))))
                a[c, c - 1] = (- Sr * pr) / dr
        elif b[c, 0] <= TaR and b[c, 0] >= TsR:
            pr = float(kr * (1 + (1.5 * c * 25 * 10 ** (-6))) / (1 + (1.5 * b[c, 0] * 10 ** (-3))))
            a[c, c] = float(((2 * Sr * pr ) + dr1) / dr1)
            a[c, c + 1] = (- Sr * pr)/ dr1
            if c > 0:
                pr = float(kr * (1 + (1.5 * c * 25 * 10 ** (-6))) / (1 + (1.5 * b[c, 0] * 10 ** (-3))))
                a[c, c - 1] = (- Sr * pr) / dr1
        elif b[c, 0] < TsR:
            pr = float(kr * (1 + (1.5 * c * 25 * 10 ** (-6))) / (1 + (1.5 * b[c, 0] * 10 ** (-3))))
            a[c, c] = float(1 + (2 * S * pr))
            a[c, c + 1] = -S * pr
            if c > 0:
                pr = float(kr * (1 + (1.5 * c * 25 * 10 ** (-6))) / (1 + (1.5 * b[c, 0] * 10 ** (-3))))
                a[c, c - 1] = -S * pr

            # Neuman boundary conditions for media change
            # between rock and magma.
    pr1 = float(kr * (1 + (1.5 * 799 * 25 * 10 ** (-6))) / (1 + (1.5 * b[Lr-1, 0] * 10 ** (-3))))
   # pr3 = float(km * (1 + (1.5 * (Lr+1) * 25 * 10 ** (-6))) / (1 + (1.5 * b[Lr+1, 0] * 10 ** (-3))))

    a[Lr, Lr - 1] = -pr1
    a[Lr, Lr] = pr1 + km
    a[Lr, Lr + 1] = -km

    #  Heat transfer  equations for  Magma injections  with phase change possibility.


    for k in range(Lr + 1, Lr + Lm, 1):
        if b[k, 0] > TL:
           # pr = float(km * (1 + (1.5 * k * 25 * 10 ** (-6))) / (1 + (1.5 * b[k, 0] * 10 ** (-3))))
            a[k, k] = float(1 + (2 * S1 ))
            a[k, k + 1] = -S1
            a[k, k - 1] = -S1
        elif b[k, 0] <= TL and b[k, 0] >= Ta:
           # pr = float(km * (1 + (1.5 * k * 25 * 10 ** (-6))) / (1 + (1.5 * b[k, 0] * 10 ** (-3))))
            a[k, k - 1] = (- S2 ) / d
            a[k, k] = float(((2 * S2 ) + d) / d)
            a[k, k + 1] = (- S2 ) / d
        elif b[k, 0] <= Ta and b[k, 0] >= Ts:
           # pr = float(km * (1 + (1.5 * k * 25 * 10 ** (-6))) / (1 + (1.5 * b[k, 0] * 10 ** (-3))))
            a[k, k - 1] = (- S2 ) / d1
            a[k, k] = float(((2 * S2 ) + d1) / d1)
            a[k, k + 1] = (- S2 ) / d1
        elif b[k, 0] < Ts:
           # pr = float(km * (1 + (1.5 * k * 25 * 10 ** (-6))) / (1 + (1.5 * b[k, 0] * 10 ** (-3))))
            a[k, k] = float(1 + (2 * S1  ))
            a[k, k + 1] = -S1
            a[k, k - 1] = -S1

    # Neuman boundary conditions for media change

    a[Lr + Lm, Lr + Lm - 1] = -km
    a[Lr + Lm, Lr + Lm] = kl + km
    a[Lr + Lm, Lr + Lm + 1] = -kl

    #  Heat transfer equations for lower with phase change possibility.

    for c in range(Lr + Lm + 1, n - 1, 1):
        if b[c, 0] > TLc:
            a[c, c] = float(1 + (2 * Sl))
            a[c, c + 1] = -Sl
            if c > 0:
                a[c, c - 1] = -Sl
        elif b[c, 0] <= TLc and b[c, 0] >= Tac:
            a[c, c] = float(((2 * Srl) + drl) / drl)
            a[c, c + 1] = - Srl / drl
            if c > 0:
                a[c, c - 1] = - Srl / drl
        elif b[c, 0] <= Tac and b[c, 0] >= Tsc:
            a[c, c] = float(((2 * Srl) + dr1l) / dr1l)
            a[c, c + 1] = - Srl / dr1l
            if c > 0:
                a[c, c - 1] = - Srl / dr1l
        elif b[c, 0] < Tsc:
            a[c, c] = float(1 + (2 * Sl))
            a[c, c + 1] = -Sl
            if c > 0:
                a[c, c - 1] = -Sl

                # isolated boundary conditios
    if b[n - 1, 0] > TLc:
        a[n - 1, n - 1] = float(1 + (2 * Sl))
        a[n - 1, n - 2] = -(2 * Sl)
    elif b[n - 1, 0] <= TLc and b[n - 1, 0] >= Tac:
        a[n - 1, n - 2] = -(2 * Srl) / drl
        a[n - 1, n - 1] = ((2 * Srl) + drl) / drl
    elif b[n - 1, 0] <= Tac and b[n - 1, 0] >= Tsc:
        a[n - 1, n - 2] = -(2 * Srl) / dr1l
        a[n - 1, n - 1] = ((2 * Srl) + dr1l) / dr1l
    elif b[n - 1, 0] < Tsc:
        a[n - 1, n - 1] = float(1 + (2 * Sl))
        a[n - 1, n - 2] = -(2 * Sl)

    # Convert a matrix in format csr
    b[0, 0] = b[0, 0] + C[0, 0]
    b[Lr] = float(0.0)
    b[Lr + Lm] = float(0.0)

#flujo de  calor
    if b[n - 1, 0] > TLc:
        b[n - 1, 0] = b[n - 1, 0] + (2 * Sl * dx * 17 * 10 ** (-3))
    elif b[n - 1, 0] <= TLc and b[n - 1, 0] >= Tac:
        b[n - 1, 0] = b[n - 1, 0] + (2 * (Srl / drl) * dx * 17 * 10 ** (-3))
    elif b[n - 1, 0] <= Tac and b[n - 1, 0] >= Tsc:
        b[n - 1, 0] = b[n - 1, 0] + (2 * (Srl / dr1l) * dx * 17 * 10 ** (-3))
    elif b[n - 1, 0] < Tsc:
        b[n - 1, 0] = b[n - 1, 0] + (2 * Sl * dx * 17 * 10 ** (-3))

    A = sparse.csr_matrix(a)
    B = np.transpose(np.mat(sparse.linalg.spsolve(A, b)))  # System AX=b solver.
    b = np.array(B)
    p = b[Lr - 2]  # Data appended
    p2 = b[Lr + Lm - 1]
    dt1 = dt1 + dt  # Total time accumulated after in each step

    # Melting fraction from crust Cr1 melting fraction

    if b[Lr - 1] > TLR:
        Cr2 = 1
    elif b[Lr - 1] < TsR:
        Cr2 = 0
    elif b[Lr - 1] <= TLR and b[Lr - 7] >= TaR:
        Cr2 = ((1.75 * 10 ** (-3)) * b[Lr - 1]) - float(1.017)
        # Cr1 = ((6.36 * 10 ** (-3)) * b[Lr - 1]) - float(5.17)
    elif b[Lr - 1] <= TaR and b[Lr - 1] >= TsR:
        Cr2 = ((6.36 * 10 ** (-3)) * b[Lr - 1]) - float(5.17)
        print (dt1)



    # Geotherm data for different times

    if j == 1601:
        w = np.zeros((n, 1))
        w = copy.copy(b)
        h1 = float(r / n)
        x1 = np.arange(0, r, h1)
        dtw = dt1 / (31.6 * 10 ** 12)
        print ("este es 1601", dtw)

        fa = np.zeros((n, 1))

        #####################################################################################
        for li in range(0, Lr, 1):

            if b[li, 0] > TLR:
                fa[li, 0] = 1
            elif b[li, 0] < TsR:
                fa[li, 0] = 0
            elif b[li, 0] <= TLR and b[li, 0] >= TaR:
                fa[li, 0] = ((1.75 * 10 ** (-3)) * b[li, 0]) - float(1.017)
            elif b[li, 0] <= TaR and b[li, 0] >= TsR:
                fa[li, 0] = ((6.36 * 10 ** (-3)) * b[li, 0]) - float(5.17)

        for ip in range(Lr + 1, Lr + Lm, 1):  # first magma injection temperature.
            if b[ip, 0] > TL:
                fa[ip, 0] = 1
            elif b[ip, 0] < Ts:
                fa[ip, 0] = 0
            elif TL >= b[ip, 0] >= Ta:
                fa[ip, 0] = ((3.25 * 10 ** (-3)) * (b[ip, 0] - TL)) + 1
            elif Ta >= b[ip, 0] >= Ts:
                fa[ip, 0] = (Xa / (Ta - Ts)) * (b[ip, 0] - Ts)

        for lo in range(Lr + Lm + 1, n, 1):
            if b[lo, 0] > TLc:
                fa[lo, 0] = 1
            elif b[lo, 0] < Tsc:
                fa[lo, 0] = 0
            elif b[lo, 0] <= TLc and b[lo, 0] >= Tac:
                fa[lo, 0] = ((3.23 * 10 ** (-3)) * b[lo, 0]) - float(2.8869)
            elif b[lo, 0] <= Tac and b[lo, 0] >= Tsc:
                fa[lo, 0] = ((1.83 * 10 ** (-3)) * b[lo, 0]) - float(1.50)
        print ("aqui va fa")
        print (fa)


    elif j == 800:
        l = np.zeros((n, 1))
        l = copy.copy(b)
        h2 = float(r / n)
        x2 = np.arange(0, r, h2)
        dtl = dt1 / (31.6 * 10 ** 12)
        print ("este es 800", dtl)

        fo = np.zeros((n, 1))

        for li in range(0, Lr, 1):

            if b[li, 0] > TLR:
                fo[li, 0] = 1
            elif b[li, 0] < TsR:
                fo[li, 0] = 0
            elif b[li, 0] <= TLR and b[li, 0] >= TaR:
                fo[li, 0] = ((1.75 * 10 ** (-3)) * b[li, 0]) - float(1.017)
            elif b[li, 0] <= TaR and b[li, 0] >= TsR:
                fo[li, 0] = ((6.36 * 10 ** (-3)) * b[li, 0]) - float(5.17)

        for ip in range(Lr + 1, Lr + Lm, 1):  # first magma injection temperature.
            if b[ip, 0] > TL:
                fo[ip, 0] = 1
            elif b[ip, 0] < Ts:
                fo[ip, 0] = 0
            elif TL >= b[ip, 0] >= Ta:
                fo[ip, 0] = ((3.25 * 10 ** (-3)) * (b[ip, 0] - TL)) + 1
            elif Ta >= b[ip, 0] >= Ts:
                fo[ip, 0] = (Xa / (Ta - Ts)) * (b[ip, 0] - Ts)

        for lo in range(Lr + Lm + 1, n, 1):
            if b[lo, 0] > TLc:
                fo[lo, 0] = 1
            elif b[lo, 0] < Tsc:
                fo[lo, 0] = 0
            elif b[lo, 0] <= TLc and b[lo, 0] >= Tac:
                fo[lo, 0] = ((3.23 * 10 ** (-3)) * b[lo, 0]) - float(2.8869)
            elif b[lo, 0] <= Tac and b[lo, 0] >= Tsc:
                fo[lo, 0] = ((1.83 * 10 ** (-3)) * b[lo, 0]) - float(1.50)
        print ("aqui va fa")
        print (fo)

    elif j == 2402:
        v = np.zeros((n, 1))
        v = copy.copy(b)
        h3 = float(r / n)
        x3 = np.arange(0, r, h3)
        dtv = dt1 / (31.6 * 10 ** 12)
        print ("este es 2402", dtv)

        fu = np.zeros((n, 1))

        for li in range(0, Lr, 1):

            if b[li, 0] > TLR:
                fu[li, 0] = 1
            elif b[li, 0] < TsR:
                fu[li, 0] = 0
            elif b[li, 0] <= TLR and b[li, 0] >= TaR:
                fu[li, 0] = ((1.75 * 10 ** (-3)) * b[li, 0]) - float(1.017)
            elif b[li, 0] <= TaR and b[li, 0] >= TsR:
                fu[li, 0] = ((6.36 * 10 ** (-3)) * b[li, 0]) - float(5.17)

        for ip in range(Lr + 1, Lr + Lm, 1):  # first magma injection temperature.
            if b[ip, 0] > TL:
                fu[ip, 0] = 1
            elif b[ip, 0] < Ts:
                fu[ip, 0] = 0
            elif TL >= b[ip, 0] >= Ta:
                fu[ip, 0] = ((3.25 * 10 ** (-3)) * (b[ip, 0] - TL)) + 1
            elif Ta >= b[ip, 0] >= Ts:
                fu[ip, 0] = (Xa / (Ta - Ts)) * (b[ip, 0] - Ts)

        for lo in range(Lr + Lm + 1, n, 1):
            if b[lo, 0] > TLc:
                fu[lo, 0] = 1
            elif b[lo, 0] < Tsc:
                fu[lo, 0] = 0
            elif b[lo, 0] <= TLc and b[lo, 0] >= Tac:
                fu[lo, 0] = ((3.23 * 10 ** (-3)) * b[lo, 0]) - float(2.8869)
            elif b[lo, 0] <= Tac and b[lo, 0] >= Tsc:
                fu[lo, 0] = ((1.83 * 10 ** (-3)) * b[lo, 0]) - float(1.50)
        print ("aqui va fa")
        print (fu)






    # f1 is de melting fraction for the first magma injection
    if p2 > TL:
        f1 = 1
    elif p2 < Ts:
        f1 = 0
    elif TL >= p2 >= Ta:
        f1 = ((3.25 * 10 ** (-3)) * (p2 - TL)) + 1
    elif Ta >= p2 >= Ts:
        f1 = (Xa / (Ta - Ts)) * (p2 - Ts)

    data.append(p2)
    X1.append(dt1 / (31.6 * 10 ** 12))
    F1.append(f1)

    if j == DT - 1:
        na = copy.copy(b)

    if j == DT + 1:
        ro = copy.copy(b)


    # j % DT == 0, if the residue of j/Dt is equal to 0  there is a magma injection
    # the size of the system change n=n+LI and the magma volume change Lm=Lm+LI

    if j % DT == 0 and u < 320:

        # print n
        n = n + LI
        Lm = Lm + LI
        a = np.zeros((n, n))
        o = np.zeros((n, 1))

        for y in range(0, Lr, 1):
            o[y] = b[y]
        for m in range(1, n, 1):
            o[Lr + LI + m] = b[Lr + m]
            if Lr + LI + m == n - 1:
                break
        for z in range(Lr + 1, Lr + LI + 1, 1):  # Temperature of new injection
            o[z] = float(1302.0)

        b = np.zeros((n, n))
        b = copy.copy(o)
        r += 50.0

        u += 1  # Magma injection counting

    if u >= 50:
        if j % DT == 0:
            dT += 2
            p3 = b[Lr + 1 + dT]  # if there's a magma injection,
            # keep the correct matrix point.
        else:
            p3 = b[Lr + 1 + dT]
        # f2 is the melting fraction for sill 50
        if p3 > TL:
            f2 = 1
        elif p3 < Ts:
            f2 = 0
        elif TL >= p3 >= Ta:
            f2 = ((3.25 * 10 ** (-3)) * (p3 - TL)) + 1
        elif Ta >= p3 >= Ts:
            f2 = (Xa / (Ta - Ts)) * (p3 - Ts)

        data2.append(p3)
        X2.append(dt1 / (31.6 * 10 ** 12))
        F2.append(f2)

    if u >= 100:
        if j % DT == 0:
            dT1 += 2
            p4 = b[Lr + 1 + dT1]
        else:
            p4 = b[Lr + 1 + dT1]

        if p4 > TL:
            f3 = 1
        elif p4 < Ts:
            f3 = 0
        elif TL >= p4 >= Ta:
            f3 = ((3.25 * 10 ** (-3)) * (p4 - TL)) + 1
        elif Ta >= p4 >= Ts:
            f3 = (Xa / (Ta - Ts)) * (p4 - Ts)

        data3.append(p4)
        X3.append(dt1 / (31.6 * 10 ** 12))
        F3.append(f3)

    if u >= 150:
        if j % DT == 0:
            dT2 += 2
            p5 = b[Lr + 1 + dT2]
        else:
            p5 = b[Lr + 1 + dT2]

        if p5 > TL:
            f4 = 1
        elif p5 < Ts:
            f4 = 0
        elif TL >= p5 >= Ta:
            f4 = ((3.25 * 10 ** (-3)) * (p5 - TL)) + 1
        elif Ta >= p5 >= Ts:
            f4 = (Xa / (Ta - Ts)) * (p5 - Ts)

        data4.append(p5)
        X4.append(dt1 / (31.6 * 10 ** 12))
        F4.append(f4)

    if u >= 200:
        if j % DT == 0:
            dT3 += 2
            p6 = b[Lr + 1 + dT3]
        else:
            p6 = b[Lr + 1 + dT3]

        if p6 > TL:
            f5 = 1
        elif p6 < Ts:
            f5 = 0
        elif p6 <= TL and p6 >= Ta:
            f5 = ((3.25 * 10 ** (-3)) * (p6 - TL)) + 1
        elif p6 <= Ta and p6 >= Ts:
            f5 = (Xa / (Ta - Ts)) * (p6 - Ts)

        data5.append(p6)
        X5.append(dt1 / (31.6 * 10 ** 12))
        F5.append(f5)

    if u >= 250:
        if j % DT == 0:
            dT4 += 2
            p7 = b[Lr + 1 + dT4]
        else:
            p7 = b[Lr + 1 + dT4]

        if p7 > TL:
            f6 = 1
        elif p7 < Ts:
            f6 = 0
        elif p7 <= TL and p7 >= Ta:
            f6 = ((3.25 * 10 ** (-3)) * (p7 - TL)) + 1
        elif p7 <= Ta and p7 >= Ts:
            f6 = (Xa / (Ta - Ts)) * (p7 - Ts)

        data6.append(p7)
        X6.append(dt1 / (31.6 * 10 ** 12))
        F6.append(f6)

    if u >= 300:
        if j % DT == 0:
            dT5 += 2
            p8 = b[Lr + 1 + dT5]
        else:
            p8 = b[Lr + 1 + dT5]

        if p8 > TL:
            f7 = 1
        elif p8 < Ts:
            f7 = 0
        elif p8 <= TL and p8 >= Ta:
            f7 = ((3.25 * 10 ** (-3)) * (p8 - TL)) + 1
        elif p8 <= Ta and p8 >= Ts:
            f7 = (Xa / (Ta - Ts)) * (p8 - Ts)

        data7.append(p8)
        X7.append(dt1 / (31.6 * 10 ** 12))
        F7.append(f7)

print ("tiempo", dt1 / (31.6 * 10 ** 12))
# Melting fraction for crust
for y in range(0, Lr, 1):
    if b[y] > TLR:
        Cr1 = 1
    elif b[y] < TsR:
        Cr1 = 0
    elif b[y] <= TLR and b[y] >= TaR:
        Cr1 = ((1.75 * 10 ** (-3)) * b[y]) - float(1.017)
    elif b[y] <= TaR and b[y] >= TsR:
        Cr1 = ((6.36 * 10 ** (-3)) * b[y]) - float(5.17)

    crust1.append(Cr1)

# melting fraction along geotherm

fi = np.zeros((n, 1))

#####################################################################################
for li in range(0, Lr, 1):

    if b[li, 0] > TLR:
        fi[li, 0] = 1
    elif b[li, 0] < TsR:
        fi[li, 0] = 0
    elif b[li, 0] <= TLR and b[li, 0] >= TaR:
        fi[li, 0] = ((1.75 * 10 ** (-3)) * b[li, 0]) - float(1.017)
    elif b[li, 0] <= TaR and b[li, 0] >= TsR:
        fi[li, 0] = ((6.36 * 10 ** (-3)) * b[li, 0]) - float(5.17)

for ip in range(Lr + 1, Lr + Lm, 1):  # first magma injection temperature.
    if b[ip, 0] > TL:
        fi[ip, 0] = 1
    elif b[ip, 0] < Ts:
        fi[ip, 0] = 0
    elif TL >= b[ip, 0] >= Ta:
        fi[ip, 0] = ((3.25 * 10 ** (-3)) * (b[ip, 0] - TL)) + 1
    elif Ta >= b[ip, 0] >= Ts:
        fi[ip, 0] = (Xa / (Ta - Ts)) * (b[ip, 0] - Ts)

for lo in range(Lr + Lm + 1, n, 1):
    if b[lo, 0] > TLc:
        fi[lo, 0] = 1
    elif b[lo, 0] < Tsc:
        fi[lo, 0] = 0
    elif b[lo, 0] <= TLc and b[lo, 0] >= Tac:
        fi[lo, 0] = ((3.23 * 10 ** (-3)) * b[lo, 0]) - float(2.8869)
    elif b[lo, 0] <= Tac and b[lo, 0] >= Tsc:
        fi[lo, 0] = ((1.83 * 10 ** (-3)) * b[lo, 0]) - float(1.50)
print ("aqui va fa")
print (fi)
####################################################################################3


h1 = float(20000.0 / (y + 1))
x4 = np.arange(0, 20000.0, h1)

# Ti1.append(dt1 / (31.6 * 10 ** 12))  # tiempo
# Tcr1.append(b[Lr - 1])  # temperatura

h = float(r / n)
x = np.arange(0, r, h)

geo = np.arange(0, 800, 20)
x5 = np.arange(0, 40000.0, 1000)

fig = plt.figure()
ax = plt.subplot(111)

plt.figure(1)
plt.gca().invert_yaxis()

ax.plot(geo, x5, label='0 Ma')
ax.plot(l, x2, label='0.79 Ma')
ax.plot(w, x1, label='1.59 Ma')
ax.plot(v, x3, label='2.39 Ma')
ax.plot(b, x, label='3.19 Ma')

ax.legend(loc=3)

ax.grid(color='black', alpha=0.7, linestyle=':', linewidth=0.4)

# plt.plot(b, x, 'b', w, x1, 'm', l, x2, 'g', v, x3, 'k',geo,x5, 'r')


# plt.axvline(x=15500, ymin=0, ymax = 900,linestyle='dashed', linewidth=1, color='b')
# plt.axvline(x=15000, ymin=0, ymax=900, linestyle='dashed', linewidth=1, color='r')
plt.ylabel(r'$ \mathrm{Profundidade (m)}$', fontsize=20)
plt.xlabel(r'$ \mathrm{Temperatura (C)}$', fontsize=20)
# plt.title('Distribucao de Temperatura Final')

fig = plt.figure()
ax = fig.add_subplot(111)
# Figures
plt.figure(2)

plt.plot(X1, F1, label='1')
plt.plot(X2, F2, label='50')
plt.plot(X3, F3, label='100')
plt.plot(X4, F4, label='150')
plt.plot(X5, F5, label='200')
plt.plot(X6, F6, label='250')
plt.plot(X7, F7, label='300')

ax.legend(loc=2)

ax.text(0.1, 3.5, r'Rhyolite', fontsize=15)
plt.axhline(y=0.21, xmin=0, linestyle='dashed', linewidth=1, color='r')
plt.axhline(y=0.425, xmin=0, linestyle='dashed', linewidth=1, color='b')
plt.axhline(y=0.650, xmin=0, linestyle='dashed', linewidth=1, color='g')
ax.grid(color='black', alpha=0.7, linestyle=':', linewidth=0.4)

#ax.text(0.5, 0.03, r'1', fontsize=15)
#ax.text(0.45, 0.17, r'50', fontsize=15)
#ax.text(0.9, 0.23, r'100', fontsize=15)
#ax.text(1.32, 0.25, r'150', fontsize=15)
#ax.text(1.95, 0.29, r'200', fontsize=15)
#ax.text(2.45, 0.39, r'250', fontsize=15)
#ax.text(2.9, 0.42, r'300', fontsize=15)
plt.xlabel(r'$ \mathrm{Tempo \;(Ma)}$', fontsize=20)
plt.ylabel(r'$ \mathrm{Fra c\c{} \~ao \; de \; fus \~a o}$', fontsize=20)
plt.title(r'$ \mathrm{ }$')


fig = plt.figure()
ax = fig.add_subplot(111)

plt.figure(3)
plt.plot(X1, data, label='1')
plt.plot(X2, data2,label='50')
plt.plot(X3, data3, label='100')
plt.plot(X4, data4, label='150')
plt.plot(X5, data5, label='200')
plt.plot(X6, data6, label='250')
plt.plot(X7, data7, label='300')

ax.legend(loc=4)

plt.axhline(y=720, xmin=0, linestyle='dashed', linewidth=1, color='r')
ax.text(1.5, 680, r'Solidus', fontsize=13)
ax.grid(color='black', alpha=0.7, linestyle=':', linewidth=0.4)
#ax.text(0.25, 700, r'1', fontsize=15)
#ax.text(0.45, 950, r'50', fontsize=15)
#ax.text(0.9, 1020, r'100', fontsize=15)
#ax.text(1.32, 1050, r'150', fontsize=15)
#ax.text(1.95, 1100, r'200', fontsize=15)
#ax.text(2.45, 1120, r'250', fontsize=15)
#ax.text(2.9, 1150, r'300', fontsize=15)

plt.xlabel(r'$ \mathrm{Tempo \;(Ma)}$', fontsize=20)
plt.ylabel(r'$ \mathrm{Temperatura (C)}$', fontsize=20)
# plt.title('Distribucao da temperatura para um ponto da rocha')

fig = plt.figure()
ax = fig.add_subplot(111)

plt.figure(4)
plt.gca().invert_yaxis()


ax.plot(fo, x2, label='0.79 Ma')
ax.plot(fa, x1, label='1.59 Ma')
ax.plot(fu, x3, label='2.39 Ma')
ax.plot(fi, x, label='3.19 Ma')

ax.legend(loc=4)

ax.grid(color='black', alpha=0.7, linestyle=':', linewidth=0.4)

# plt.plot(b, x, 'b', w, x1, 'm', l, x2, 'g', v, x3, 'k',geo,x5, 'r')


# plt.axvline(x=15500, ymin=0, ymax = 900,linestyle='dashed', linewidth=1, color='b')
# plt.axvline(x=15000, ymin=0, ymax=900, linestyle='dashed', linewidth=1, color='r')
plt.ylabel(r'$ \mathrm{Profundidade (m)}$', fontsize=20)
plt.xlabel(r'$\mathrm{Fra c\c{} \~ao \; de \; fus \~a o}$', fontsize=20)
# plt.title('Distribucao de Temperatura Final')



plt.show()