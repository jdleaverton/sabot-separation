# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 13:35:43 2021

@author: jleaver1
"""

import matplotlib.pyplot as plt
import numpy as np
from math import cosh, sinh


x = np.array([0, 0.038, 0.076, 0.114, 0.152])
X = np.linspace(0, 0.152, 100)

smallSteel = np.array([49.2, 29.4, 22.1, 20.2, 20])
smallSteelN = (smallSteel - 19.7) / (49.2 - 19.7) 
smallAlum = np.array([49.4, 35.7, 27.3, 24.4, 22.8])
smallAlumN = (smallAlum - 19.7) / (49.4 - 19.7) 
bigSteel = np.array([50.9, 32.4, 24, 21.1, 20.5])
bigSteelN = (bigSteel - 19.8) / (50.9 - 19.8) 
bigAlum = np.array([50.5, 33.3, 30, 47.8, 25.1])
bigAlumN = (bigAlum - 19.8) / (50.5 - 19.8) 


def ss(x):
    m = 28.17
    L = 0.152
    k = 0.0263
    h = 152.46
    T_b = 49.2
    T_inf = 19.7
    theta_b = T_b - T_inf
    return ((cosh(m * (L-x)) + (h / (m*k)) * sinh(m * (L-x) )) / (cosh(m * L) + (h / (m*k)) * sinh(m * L)))


def sa(x):
    m = 16.47
    L = 0.152
    k = 0.0263
    h = 152.46
    T_b = 49.4
    T_inf = 19.7
    theta_b = T_b - T_inf
    return ((cosh(m * (L-x)) + (h / (m*k)) * sinh(m * (L-x) )) / (cosh(m * L) + (h / (m*k)) * sinh(m * L)))

def ls(x):
    m = 23.30
    L = 0.152
    k = 0.0263
    h = 156.37
    T_b = 50.9
    T_inf = 19.8
    theta_b = T_b - T_inf
    return ((cosh(m * (L-x)) + (h / (m*k)) * sinh(m * (L-x) )) / (cosh(m * L) + (h / (m*k)) * sinh(m * L))) 

def la(x):
    m = 13.62
    L = 0.152
    k = 0.0263
    h = 156.37
    T_b = 50.5
    T_inf = 19.8
    theta_b = T_b - T_inf
    return ((cosh(m * (L-x)) + (h / (m*k)) * sinh(m * (L-x) )) / (cosh(m * L) + (h / (m*k)) * sinh(m * L))) 


fig, ax = plt.subplots(dpi = 300)
plt.scatter(x, bigSteelN, label='Large Steel')
plt.scatter(x, smallSteelN, label='Small Steel')
plt.scatter(x, bigAlumN, label='Large Aluminum')
plt.scatter(x, smallAlumN, label='Small Aluminum')
ssf = np.vectorize(ss)
#plt.plot(X, ssf(X), c='darkorange')
saf = np.vectorize(sa)
#plt.plot(X, saf(X), c='red')
lsf = np.vectorize(ls)
#plt.plot(X, lsf(X), c='tab:blue')
laf = np.vectorize(la)
#plt.plot(X, laf(X), c='tab:green')
plt.title('Measured axial dimensionless temperature distributions along the fin')
ax.set_xlabel('Distance from fin base (m)')
ax.set_ylabel('Dimensionless Temperature')
plt.legend()

fig, axs = plt.subplots(dpi = 300, nrows=2, ncols=2, sharex=True, sharey=True, figsize=[6, 6])

fig.text(0.5, 0.04, 'Distance from fin base (m)', ha='center')
fig.text(0.04, 0.5, 'Dimensionless Temperature', va='center', rotation='vertical')
plt.suptitle('Measured axial dimensionless temperature distributions with analytical solutions')
ax = axs[0, 0]
ax.scatter(x, smallSteelN, label='Small Steel')
ssf = np.vectorize(ss)
ax.plot(X, ssf(X))
ax.set_title('Small Steel')

ax = axs[0, 1]
ax.scatter(x, bigSteelN, label='Large Steel')
lsf = np.vectorize(ls)
ax.plot(X, lsf(X))
ax.set_title('Large Steel')

ax = axs[1, 0]
ax.scatter(x, smallAlumN, label='Small Aluminum')
saf = np.vectorize(sa)
ax.plot(X, saf(X))
ax.set_title('Small Aluminum')

ax = axs[1, 1]
ax.scatter(x, bigAlumN, label='Large Aluminum')
laf = np.vectorize(la)
ax.plot(X, laf(X))
ax.set_title('Large Aluminum')



x2 = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
actualT = np.array([51.7, 42.2, 36.4, 32.5, 29.9, 28.8, 28.4])
normalizedT = (actualT - 19.8) / (51.7 - 19.8)
fig, ax = plt.subplots(dpi = 300)
plt.scatter(x2, normalizedT, label='Measured data')
X2 = np.linspace(0, 0.3, 100)
def convection(x):
    m = 6.675
    L = 0.3
    k = 0.0263
    return ((cosh(m * (L - x))) / (cosh(m * L)))

convectionf = np.vectorize(convection)
plt.plot(X2, convectionf(X2), label='Analytical solution')

plt.title('Measured axial dimensionless temperature distributions with analytical solution')
ax.set_xlabel('Distance from fin base (m)')
ax.set_ylabel('Dimensionless Temperature')
plt.legend()
