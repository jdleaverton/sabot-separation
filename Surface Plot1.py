# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 20:04:55 2021

@author: JD
"""

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap 
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import cm
from scipy.interpolate import griddata
from math import sqrt, exp

labelled_data = pd.read_csv('data.csv')
V = labelled_data.iloc[:,0].values
P = labelled_data.iloc[:,1].values
D = labelled_data.iloc[:,2].values
S = labelled_data.iloc[:,3].values



data = np.zeros([V.shape[0], 3])

for i in range(len(data)):
    for j in range(3):
        if j == 0:
            data[i, j] = V[i]
        elif j == 1:
            data[i, j] = P[i]
        elif j == 2:
            data[i, j] = D[i]

data4mm = np.array([[6484.108046,	100,	 1.356073197],
                    [1971.13232,	200,	 1.709645283],
                    [2345.6,        199,    1.66176808],
                    [6369.1,        85,     1.093763538],
                    [5527.0,        90,     1.35479075],
                    [6069.7,        90,     1.132869464],
                    [2617.6,        149,    1.406661296]
                     ],dtype=float)
proposed4mm = np.array([[2000, 125],
                  [2000, 275],
                  [2250, 160],
                  [2250, 250],
                  [2500, 225],
                  [2500, 115],
                  [2750, 175],
                  [2750, 250],
                  [3000, 125],
                  [3000, 215],
                  [3250, 175],
                  [3250, 250],
                  [3500, 140],
                  [3500, 210],
                  [3750, 115],
                  [3750, 180],
                  [4000, 200],
                  [4000, 100],
                  [4250, 125],
                  [4250, 175],
                  [4500, 210],
                  [4500, 150],
                  [4750, 180],
                  [4750, 110],
                  [5000, 85],
                  [5000, 140],
                  [5250, 160],
                  [5250, 115],
                  [5500, 185],
                  [5500, 140],
                  [5750, 165],
                  [5750, 110],
                  [6000, 125],
                  [6000, 150],
                  [6250, 115],
                  [6250, 80],
                  [6500, 130],
                  [6500, 160],
                  [6750, 100],
                  [6750, 80]  
                  ], dtype=float)
proposed10mm = np.array([
                  [2250, 250],
                  [2500, 100],
                  [3000, 150],
                  [3500, 175],
                  [4000, 175],
                  [4500, 150],
                  [5000, 90],
                  [5500, 75],
                  [6000, 150],
                  [6500, 120]], dtype=float)

failed = np.array([[6060.6, 175],
                  [4908.4, 250],
                  [4676.4, 210],
                  [5207.0, 252],
                  [5495.5, 175],
                  [4164.9, 200],
                  [6035.2, 200],
                  [5378.7, 250],
                  [5470.3, 252],
                  [4789.2, 250],
                  [6527.0, 200],
                  [6107.1, 200],
                  [6141.7, 201],
                  [5855.0, 90]], dtype=float)



labelled_data = pd.read_csv('data.csv')
# 4th order regression
V = labelled_data.iloc[:,0].values
P = labelled_data.iloc[:,1].values
D = labelled_data.iloc[:,2].values
S = labelled_data.iloc[:,3].values

def regression():
    S_poly = np.zeros((36, 7))
    S_poly[:, 0] = np.ones(1) #M0D0
    S_poly[:, 1] = V*1 #M1D0
    S_poly[:, 2] = V*V*1 #M2D0
    S_poly[:, 3] = 1*P #M0D1
    S_poly[:, 4] = V*P #M1D1
    S_poly[:, 5] = 1*P #M0D2
    S_poly[:, 6] = 1*S #M0D2
    
    
    S_opt = S_poly[:, [0, 4, 5, 6]]
    regressor_OLS = sm.OLS(endog = D, exog = S_opt).fit()
    print(regressor_OLS.summary())

regression()

def createSurface():
    # regular grid covering the domain of the data
    X,Y = np.meshgrid(np.arange(2000, 7000, 250), np.arange(0, 350, 25))
    XX = X.flatten()
    YY = Y.flatten()
    
    
    order = 2    # 1: linear, 2: quadratic
    if order == 1:
        # best-fit linear plane
        A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
        C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients
        
        # evaluate it on grid
        Z = C[0]*X + C[1]*Y + C[2]
        
        # or expressed using matrix/vector product
        #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)
    
    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
        C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
        
        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
    
createSurface()

# create custom colormap
def custom_div_cmap(numcolors, name,
                    mincol, minmidcol, midcol, midmaxcol, maxcol):

    
    
    cmap = LinearSegmentedColormap.from_list(name=name, 
                                             colors =[mincol, minmidcol, midcol, midmaxcol, maxcol],
                                             N=numcolors, gamma=1.1)
    return cmap
    
custom_map = custom_div_cmap(100, 'custom_map', mincol='orange', minmidcol='orange', midcol='forestgreen', midmaxcol='orange', maxcol='orange')


def mm4(v, p):
    return 0.26682 + 0.00918261 * p + 4.8841e-07 * v * p - 1.711e-05 * p**2

def mm10(v, p):
    return 0.26682 + 0.00918261 * p + 4.8841e-07 * v * p - 1.711e-05 * p**2 + 0.266455432


def topboundary10mm(v):
    return np.real(0.0142726475 * v - 29222.6768 * sqrt(0.000000000000238544328 * v**2 + 0.0000000089697571002689 * v - 0.000022906303023262) + 268.340444207)
def botboundary10mm(v):
    return np.real(0.0142726475 * v - 29222.6768 * sqrt(0.000000000000238544328 * v**2 + 0.0000000089697571002689 * v + 0.00003868969697862212) + 268.340444207)

def topboundary4mm(v):
    return np.real(0.0142726475 * v - 29222.6768 * sqrt(0.000000000000238544328 * v**2 + 0.0000000089697571002689 * v - 0.00003087651278832) + 268.340444207)
def botboundary4mm(v):
    return np.real(0.0142726475 * v - 29222.6768 * sqrt(0.000000000000238544328 * v**2 + 0.0000000089697571002689 * v + 0.00002045348721324558) + 268.340444207)

def logistic(v):
    return 1 / (1 + exp(-0.000694 * (v - 4500)))


topB10mm = np.vectorize(topboundary10mm)
botB10mm = np.vectorize(botboundary10mm)

topB4mm = np.vectorize(topboundary4mm)
botB4mm = np.vectorize(botboundary4mm)

def modified10mm(v):
    return botboundary10mm(v) * logistic(v) + topboundary10mm(v) * (1 - logistic(v))

def modified4mm(v):
    return botboundary4mm(v) * logistic(v) + topboundary4mm(v) * (1 - logistic(v))
modifiedV10mm = np.vectorize(modified10mm)
modifiedV4mm = np.vectorize(modified4mm)

sabot = 4


x = np.linspace(1500, 7000, 25)
y = np.linspace(0, 300, 25)

X, Y = np.meshgrid(x, y)

if sabot == 4:
    Z = mm4(X, Y)
    # plot points and fitted surface
    fig = plt.figure(figsize=[8, 6], dpi=300)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap=custom_map, rcount=100, shade=True, vmin=0, vmax=3, antialiased=False, alpha=0.8)
    ax.scatter(data[:,0], data[:,1], data[:,2], c='r', label='Test data points')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Backfill Pressure (Torr)')
    plt.title('Sabot separation empirical relationship')
    ax.set_zlabel('Sabot Radius (in)')
    #ax.axis('equal')
    #ax.axis('tight')
    ax.set_xlim(7000, 2000)
    ax.set_ylim(0, 300)
    plt.show()
    
    fig = plt.figure(figsize=[8, 6], dpi=300)
    plt.title(str(sabot) + ' mm Sabot')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Backfill Pressure (Torr)')
    plt.contourf(X, Y, Z, 20, cmap=custom_map, vmin=0, vmax=3, alpha = 0.8, antialiased=False)
    cb = plt.colorbar(label='Degree of Separation (inches)')
    
    cb.ax.plot([0, 3], [1.2, 1.2], 'k--')
    cb.ax.plot([0, 3], [1.95, 1.95], 'k--')
    plt.scatter(failed4mm[:, 0], failed4mm[:, 1], marker='x', color='black', label='Sabot Failure')
    plt.scatter(data4mm[:, 0], data4mm[:, 1], facecolors='none', edgecolors='black', label='Successful tests')
    plt.scatter(proposed4mm[:, 0], proposed4mm[:, 1], facecolors='blue', edgecolors='blue', label='Proposed Tests')
    xtopboundary = np.linspace(3175, 7000, 100)
    xbotboundary = np.linspace(1500, 7000, 100)
    xmodified = np.linspace(1750, 7000, 100)
    plt.plot(xtopboundary, topB4mm(xtopboundary), color='k', linestyle='--')
    plt.plot(xbotboundary, botB4mm(xbotboundary), color='k', linestyle='--')
    plt.plot(xmodified, modifiedV4mm(xtopboundary), label='Sabot stripper life optimization curve')
    x1,x2,y1,y2 = plt.axis()  
    plt.axis((x1,x2,0,300))
    plt.legend(loc='lower left')
    
    fig = plt.figure(dpi=300, figsize=[6.0, 6.0])
    ax = Axes3D(fig)
    n = 256
    m = 24
    rad = np.linspace(0, 3, m)
    a = np.linspace(0, 2 * np.pi, n)
    r, th = np.meshgrid(rad, a)
    innerRadius = 1.25 / 2
    radius = innerRadius * np.ones(n)
    radiusO = 2.1 * np.ones(n)
    radiusI = 1.2 * np.ones(n)
    
    z = r
    plt.subplot(projection="polar", title='Degree of Separation visualized on stripper plate')
    plt.contourf(th, r, z, 20, cmap=custom_map, vmin=0, vmax=3, alpha=0.8, antialiased=False)
    
    plt.plot(a, radius, linewidth=3.0, color='k')
    #plt.plot(a, radiusI, linestyle='--', color='k')
    #plt.plot(a, radiusO, linestyle='--', color='k')
    
    #plt.xlabel('Degree of Separation (inches)')
    plt.xticks([])
    plt.grid(color='k', alpha=0.5)
    plt.yticks([0, 1, 2, 3])
    plt.title('Degree of Separation visualized on stripper plate')
    
    plt.show()
    
elif sabot == 10:
    Z = mm10(X, Y)
    # plot points and fitted surface
    fig = plt.figure(figsize=[8, 6], dpi=300)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap=custom_map, rcount=100, shade=True, vmin=0, vmax=3, antialiased=False, alpha=0.8)
    ax.scatter(data[:,0], data[:,1], data[:,2], c='r', label='Test data points')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Backfill Pressure (Torr)')
    plt.title('Sabot separation empirical relationship')
    ax.set_zlabel('Sabot Radius (in)')
    #ax.axis('equal')
    #ax.axis('tight')
    ax.set_xlim(7000, 2000)
    ax.set_ylim(0, 300)
    plt.show()
    
    fig = plt.figure(figsize=[8, 6], dpi=300)
    plt.title(str(sabot) + ' mm Sabot')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Backfill Pressure (Torr)')
    plt.contourf(X, Y, Z, 20, cmap=custom_map, vmin=0, vmax=3, alpha = 0.8, antialiased=False)
    cb = plt.colorbar(label='Degree of Separation (inches)')
    cb.ax.plot([0, 3], [1.2, 1.2], 'k--')
    cb.ax.plot([0, 3], [2.1, 2.1], 'k--')
    plt.scatter(failed10mm[:, 0], failed10mm[:, 1], marker='x', color='black', label='Sabot Failure')
    plt.scatter(data10mm[:, 0], data10mm[:, 1], facecolors='none', edgecolors='black', label='Successful tests')
    plt.scatter(proposed10mm[:, 0], proposed10mm[:, 1], facecolors='blue', edgecolors='blue', label='Proposed Tests')
    xtopboundary = np.linspace(2402, 7000, 50)
    xbotboundary = np.linspace(1500, 7000, 50)
    xmodified = np.linspace(1750, 7000, 50)
    plt.plot(xtopboundary, topB10mm(xtopboundary), color='k', linestyle='--')
    plt.plot(xbotboundary, botB10mm(xbotboundary), color='k', linestyle='--')
    plt.plot(xmodified, modifiedV10mm(xtopboundary), label='Sabot stripper life optimization curve')
    plt.legend(loc='lower left')
    
    fig = plt.figure(dpi=300, figsize=[6.0, 6.0])
    ax = Axes3D(fig)
    n = 256
    m = 24
    rad = np.linspace(0, 3, m)
    a = np.linspace(0, 2 * np.pi, n)
    r, th = np.meshgrid(rad, a)
    innerRadius = 1.25 / 2
    radius = innerRadius * np.ones(n)
    radiusO = 2.1 * np.ones(n)
    radiusI = 1.2 * np.ones(n)
    
    z = r
    plt.subplot(projection="polar", title='Degree of Separation visualized on stripper plate')
    plt.contourf(th, r, z, 20, cmap=custom_map, alpha=0.8, antialiased=False)
    
    plt.plot(a, radius, linewidth=3.0, color='k')
    #plt.plot(a, radiusI, linestyle='--', color='k')
    #plt.plot(a, radiusO, linestyle='--', color='k')
    
    #plt.xlabel('Degree of Separation (inches)')
    plt.xticks([])
    plt.grid(color='k', alpha=0.5)
    plt.yticks([])
    plt.title('Degree of Separation visualized on stripper plate')
    
    plt.show()


