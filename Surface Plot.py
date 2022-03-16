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
from matplotlib import animation


labelled_data = pd.read_csv('data.csv')
V = labelled_data.iloc[:,0].values
P = labelled_data.iloc[:,1].values
D = labelled_data.iloc[:,2].values
S = labelled_data.iloc[:,3].values

data = np.zeros([V.shape[0], 3])
dataFull = np.zeros([V.shape[0], 4])
data4mm = np.zeros([V.shape[0], 3])
data10mm = np.zeros([V.shape[0], 3])

proposed4mm = np.array([[2, 15],
                        [2, 31],
                        [2, 20],
                        [3, 25],
                        [3.5, 22],
                        [3.5, 14],
                        [4, 13],
                        [4.5, 21],
                        [5, 15],
                        [5.5, 17]
                  ], dtype=float)
proposed10mm = np.array([
                  [2, 20],
                  [2, 15],
                  [3, 20],
                  [3.5, 20],
                  [4, 18],
                  [4.5, 17],
                  [5, 16],
                  [5.5, 11],
                  ], dtype=float)

for i in range(len(data)):
    for j in range(4):
        if j == 0:
            data[i, j] = V[i]
            dataFull[i, j] = V[i]
        elif j == 1:
            data[i, j] = P[i]
            dataFull[i, j] = P[i]
        elif j == 2:
            data[i, j] = D[i]
            dataFull[i, j] = D[i]
        elif j == 3:
            dataFull[i, j] = S[i]
            
for i in range(len(data)):
    for j in range(3):
        if j == 0:
            if dataFull[i, 3] == 1:
                data10mm[i, j] = V[i]
            elif dataFull[i, 3] == 0:
                data4mm[i, j] = V[i]
        elif j == 1:
            if dataFull[i, 3] == 1:
                data10mm[i, j] = P[i]
            elif dataFull[i, 3] == 0:
                data4mm[i, j] = P[i]
        elif j == 2:
            if dataFull[i, 3] == 1:
                data10mm[i, j] = D[i]
            elif dataFull[i, 3] == 0:
                data4mm[i, j] = D[i]



# failed 
failed10mm = np.array([
              [4.6764, 27.9977],
              [5.2070, 33.5972],
              [5.4955, 23.3314],
              [4.1649, 26.6645],
              [6.0352, 26.6645],
              [5.3787, 33.3306],
              [5.4703, 33.5972],
              [4.7892, 33.3306],
              [6.1417, 26.7978],
              [5.8550, 11.999]], dtype=float)

failed4mm = np.array([
              [6.5270, 26.6645],
              [6.1071, 26.6645]]
              , dtype=float)



def regression(V, P, S, D):
   
    S_poly = np.zeros((55, 7))
    S_poly[:, 0] = np.ones(1)
    S_poly[:, 1] = V*1
    S_poly[:, 2] = V*V*1
    S_poly[:, 3] = 1*P
    S_poly[:, 4] = V*P
    S_poly[:, 5] = P*P*1
    S_poly[:, 6] = 1*S*P
    S_opt = S_poly[:, [3, 4, 5, 6]]
    regressor_OLS = sm.GLS(endog = D, exog = S_opt).fit()
    print(regressor_OLS.summary())
regression(V, P, S, D)


# regular grid covering the domain of the data
X,Y = np.meshgrid(np.arange(1.5, 7.25, 0.25), np.arange(0, 41, 1))
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


# create custom colormap
def custom_div_cmap(numcolors, name,
                    mincol, minmidcol, midcol, midmaxcol, maxcol):

    
    
    cmap = LinearSegmentedColormap.from_list(name=name, 
                                             colors =[mincol, minmidcol, midcol, midmaxcol, maxcol],
                                             N=numcolors, gamma=1.1)
    return cmap

custom_map = custom_div_cmap(100, 'custom_map', mincol='orange', minmidcol='orange', midcol='forestgreen', midmaxcol='darkorange', maxcol='darkorange')





def surface4mm(v, p):
    return 2.71828182846**(np.log(p) + 0.0181021 * p * v)

def surface10mm(v, p):
    return 2.71828182846**(np.log(p) + 0.0181021 * p * v -0.142069 * p)


def mm10(v, p):
    return 9.6003 + 1.5230 * p + 0.0718 * v * p - 0.0177 * p**2 + 6.5138

def mm4(v, p):
    return 9.6003 + 1.5230 * p + 0.0718 * v * p - 0.0177 * p**2

'''4mm'''
fig = plt.figure()
ax = Axes3D(fig)
def init():
    # plot points and fitted surface
    Z = surface4mm(X, Y)
    ax.plot_surface(X, Y, Z, cmap=custom_map, rcount=100, shade=True, vmin=0, vmax=75, antialiased=False, alpha=0.8)
    ax.scatter(data4mm[:,0], data4mm[:,1], data4mm[:,2], c='r', label='Test data points')
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Backfill Pressure (kPa)')
    plt.title('4mm')
    ax.set_zlabel('Sabot Radius (mm)')
    #ax.axis('equal')
    #ax.axis('tight')
    ax.set_xlim(7, 1)
    ax.set_ylim(0, 50)
    ax.set_zlim(0, 80)
    return fig,
    

def animate(i):
    ax.view_init(elev=10, azim=i*4)
    return fig,


ani = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=50, blit=True)
fn = '4mm'
#ani.save(fn+'.mp4', writer='ffmpeg', fps=60)
ani.save(fn+'.gif',writer='imagemagick',fps=60)


'''10mm'''
fig = plt.figure()
ax = Axes3D(fig)
def init():
    # plot points and fitted surface
    Z = surface10mm(X, Y)
    ax.plot_surface(X, Y, Z, cmap=custom_map, rcount=100, shade=True, vmin=0, vmax=75, antialiased=False, alpha=0.8)
    ax.scatter(data10mm[:,0], data10mm[:,1], data10mm[:,2], c='r', label='Test data points')
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Backfill Pressure (kPa)')
    plt.title('10mm')
    ax.set_zlabel('Sabot Radius (mm)')
    #ax.axis('equal')
    #ax.axis('tight')
    ax.set_xlim(7, 1)
    ax.set_ylim(0, 50)
    return fig,
    

def animate(i):
    ax.view_init(elev=10, azim=i*4)
    return fig,


ani = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=50, blit=True)
fn = '10mm'
#ani.save(fn+'.mp4', writer='ffmpeg', fps=60)
ani.save(fn+'.gif',writer='imagemagick',fps=60)





def torr2kPa(torr):
    return torr / 7.501

def kPa2torr(kPa):
    return kPa * 7.501

def plotSurfaces():
    
    '''4mm'''
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    plt.rcParams['lines.markersize'] = 8
    levels = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    levels2 = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5, 55, 57.5, 60, 62.5, 65, 67.5, 70, 72.5, 75, 77.5, 80]
    Z4mm = surface4mm(X, Y)
    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=300, figsize=[18, 9])
    
    ax1.set_title(str(4) + ' mm Sabot')
    ax1.set_xlabel('Velocity (km/s)')
    ax1.set_ylabel('Backfill Pressure (kPa)')
    m = ax1.contourf(X, Y, Z4mm, levels2, cmap=custom_map, vmin=0, vmax=75, alpha = 0.8, antialiased=False)
    
    #cb.ax.plot([0, 3], [1.2, 1.2], 'k--')
    #cb.ax.plot([0, 3], [1.95, 1.95], 'k--')
    ax1.scatter(data4mm[:, 0], data4mm[:, 1], facecolors='none', edgecolors='black', label='Successful tests')
    ax1.scatter(failed4mm[:, 0], failed4mm[:, 1], marker='x', color='black', label='Sabot Failure')
    #ax1.scatter(proposed4mm[:, 0], proposed4mm[:, 1], facecolors='blue', edgecolors='blue', label='Proposed Tests')
    
    #xtopboundary = np.linspace(3175, 7000, 100)
    #xbotboundary = np.linspace(1500, 7000, 100)
    #xmodified = np.linspace(1750, 7000, 100)
    #plt.plot(xtopboundary, topB4mm(xtopboundary), color='k', linestyle='--')
    #plt.plot(xbotboundary, botB4mm(xbotboundary), color='k', linestyle='--')
    #plt.plot(xmodified, modifiedV4mm(xtopboundary), label='Sabot stripper life optimization curve')
    x1,x2,y1,y2 = ax1.axis()  
    ax1.axis((1.5,7,0,40))
    secax = ax1.secondary_yaxis('right', functions=(kPa2torr, torr2kPa))
    secax.set_ylabel('Backfill Pressure (Torr)')
    ax1.legend(loc='lower left')
    
    '''10mm'''
    
    Z10mm = surface10mm(X, Y)
    ax2.set_title(str(10) + ' mm Sabot')
    ax2.set_xlabel('Velocity (km/s)')
    ax2.set_ylabel('Backfill Pressure (kPa)')
    ax2.contourf(X, Y, Z10mm, levels2, cmap=custom_map, vmin=0, vmax=75, alpha = 0.8, antialiased=False)
    #cb = plt.colorbar(label='Degree of Separation (mm)', pad=0.1)
    #cb.ax.plot([0, 3], [1.2, 1.2], 'k--')
    #cb.ax.plot([0, 3], [2.1, 2.1], 'k--')
    ax2.scatter(data10mm[:, 0], data10mm[:, 1], facecolors='none', edgecolors='black', label='Successful tests')
    ax2.scatter(failed10mm[:, 0], failed10mm[:, 1], marker='x', color='black', label='Sabot Failure')
    #ax2.scatter(proposed10mm[:, 0], proposed10mm[:, 1], facecolors='blue', edgecolors='blue', label='Proposed Tests')
    ax2.axis((1.5,7,0,40))
    secax = ax2.secondary_yaxis('right', functions=(kPa2torr, torr2kPa))
    secax.set_ylabel('Backfill Pressure (Torr)')
   
    #xtopboundary = np.linspace(2402, 7000, 50)
    #xbotboundary = np.linspace(1500, 7000, 50)
    #xmodified = np.linspace(1750, 7000, 50)
    #plt.plot(xtopboundary, topB10mm(xtopboundary), color='k', linestyle='--')
    #plt.plot(xbotboundary, botB10mm(xbotboundary), color='k', linestyle='--')
    #plt.plot(xmodified, modifiedV10mm(xtopboundary), label='Sabot stripper life optimization curve')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.055, 0.05, 0.885, 0.04])
    cb = fig.colorbar(m, cax=cbar_ax, label='Degree of Separation (mm)', orientation='horizontal', drawedges=False, ticks=levels)

plotSurfaces()

def plotOthers():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    plt.rcParams['lines.markersize'] = 8
        
    '''radial'''
    
    fig, ax1 = plt.subplots(subplot_kw={'projection' : 'polar'}, dpi=300, figsize=[6.0, 6.0])
    n = 256
    m = 24
    rad = np.linspace(0, 76.2, m)
    a = np.linspace(0, 2 * np.pi, n)
    r, th = np.meshgrid(rad, a)
    innerRadius = 31.75 / 2
    radius = innerRadius * np.ones(n)
    
    z = r
    ax1.contourf(th, r, z, 20, cmap=custom_map, vmin=0, vmax=75, alpha=0.8, antialiased=False)
    
    ax1.plot(a, radius, linewidth=3.0, color='k')
    #plt.plot(a, radiusI, linestyle='--', color='k')
    #plt.plot(a, radiusO, linestyle='--', color='k')
    
    #plt.ylabel('Degree of Separation (mm)')
    
    ax1.set_rticks([25, 50, 75])
    ax1.set_xticks([])
    ax1.grid(True, color='k', alpha=0.5)
    
    ax1.set_title('Degree of Separation visualized on sabot stopping plate')
    
    
    '''3D surface'''
    '''
    fig = plt.figure(figsize=[8, 6], dpi=400)
    ax2 = Axes3D(fig)
    x = np.linspace(1, 7, 20)
    y = np.linspace(0, 50, 20)
    X, Y = np.meshgrid(x, y)
    Z = noDistinction(X, Y)
    
    # plot points and fitted surface
    ax2.plot_surface(X, Y, Z, cmap=custom_map, rcount=100, shade=True, vmin=0, vmax=75, antialiased=False, alpha=0.8)
    ax2.scatter(data[:,0], data[:,1], data[:,2], c='r', label='Test data points')
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Backfill Pressure (kPa)')
    plt.title('Sabot separation empirical surface - No geometry distinction')
    ax2.set_zlabel('Sabot Radius (mm)')
    #ax.axis('equal')
    #ax.axis('tight')
    ax2.set_xlim(7, 1)
    ax2.set_ylim(0, 50)
'''

plotOthers()