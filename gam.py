# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 16:10:25 2021

@author: jleaver1
"""

from pygam.datasets import wage
from pygam import LinearGAM, s, f, te
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd
from math import sqrt

labelled_data = pd.read_csv('data.csv')
V = labelled_data.iloc[:,0].values
P = labelled_data.iloc[:,1].values
y = labelled_data.iloc[:,2].values
S = labelled_data.iloc[:,3].values

X = np.zeros([V.shape[0], 3])
dataFull = np.zeros([V.shape[0], 4])
data4mm = np.zeros([V.shape[0], 3])
data10mm = np.zeros([V.shape[0], 3])


for i in range(len(X)):
    for j in range(4):
        if j == 0:
            X[i, j] = V[i]
            dataFull[i, j] = V[i]
        elif j == 1:
            X[i, j] = P[i]
            dataFull[i, j] = P[i]
        elif j == 2:
            dataFull[i, j] = y[i]
            X[i, j] = S[i]
        elif j == 3:
            dataFull[i, j] = S[i]
                        
for i in range(len(X)):
    
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
                data10mm[i, j] = y[i]
            elif dataFull[i, 3] == 0:
                data4mm[i, j] = y[i]



data4mmNew = data4mm[np.nonzero(data4mm)]
data4mm = data4mmNew.reshape(22, 3, order='A')

data10mmNew = data10mm[np.nonzero(data10mm)]
data10mm = data10mmNew.reshape(33, 3, order='A')

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
                [6.1071, 26.6645]], dtype=float)

failedAll = np.array([
              [4.6764, 27.9977],
              [5.2070, 33.5972],
              [5.4955, 23.3314],
              [4.1649, 26.6645],
              [6.0352, 26.6645],
              [5.3787, 33.3306],
              [5.4703, 33.5972],
              [4.7892, 33.3306],
              [6.1417, 26.7978],
              [5.8550, 11.999],
              [6.5270, 26.6645],
              [6.1071, 26.6645]], dtype=float)


gam = LinearGAM(te(1, 0) + f(2))
gam.gridsearch(X, y)

gam.summary()

## generate meshgrid for prediction plotting
n = 10000
m = int(sqrt(n))
xlin = np.linspace(1.5, 6.5, m, dtype=float)
ylin = np.linspace(10, 40, m, dtype=float)


for i in range(len(xlin)):
    for j in range(len(xlin)):
        
        if i==0 and j==0:
            prediction4mm = np.array([[xlin[i], ylin[j], 0]], dtype=float)
            prediction10mm = np.array([[xlin[i], ylin[j], 1]], dtype=float)
        else:
            prediction4mm = np.append(prediction4mm, [[xlin[i], ylin[j], 0]], axis=0)
            prediction10mm = np.append(prediction10mm, [[xlin[i], ylin[j], 1]], axis=0)

Z4mm = gam.predict(prediction4mm)
full4mm = np.column_stack((prediction4mm, Z4mm))
Z4mm = np.reshape(Z4mm, [m, m])

Z10mm = gam.predict(prediction10mm)
full10mm = np.column_stack((prediction10mm, Z10mm))
Z10mm = np.reshape(Z10mm, [m, m])

Z4mm = np.rot90(np.fliplr(Z4mm))
Z10mm = np.rot90(np.fliplr(Z10mm))

## plotting

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20
plt.rcParams['lines.markersize'] = 8

'''Prediction - 4mm'''
plt.figure()
plt.ion()
plt.rcParams['figure.figsize'] = (12, 8)
XX = gam.generate_X_grid(term=0, meshgrid=True)
Z = gam.partial_dependence(term=0, X=XX, meshgrid=True)
ax = plt.axes(projection='3d')
ax.plot_surface(XX[1], XX[0], Z4mm, cmap='viridis')
ax.scatter(data4mm[:,0], data4mm[:,1], data4mm[:,2], c='r', label='Test data points')
ax.set_title("Prediction- 4mm")
ax.set_zlim(30, 55)
plt.show()

'''Prediction - 10mm'''
plt.figure()
plt.ion()
plt.rcParams['figure.figsize'] = (12, 8)
XX = gam.generate_X_grid(term=0, meshgrid=True)
Z = gam.partial_dependence(term=0, X=XX, meshgrid=True)
ax = plt.axes(projection='3d')
ax.plot_surface(XX[1], XX[0], Z10mm, cmap='viridis')
ax.scatter(data10mm[:,0], data10mm[:,1], data10mm[:,2], c='r', label='Test data points')
ax.set_title("Prediction- 10mm")
ax.set_zlim(30, 55)


'''Partial Dependence'''
# 3D
fig = plt.figure(dpi=300, figsize=[18, 9])
fig.suptitle('Partial Dependence of Degree of Separation - Tensor')
ax = fig.add_subplot(1, 2, 1, projection='3d')
#ax.add_gridspec()
ax.set_xlabel("Velocity (km/s)")
ax.set_ylabel("Backfill Pressure (kPa)")
ax.set_zlabel("Change in Degree of Separation (mm)")
XX = gam.generate_X_grid(term=0, meshgrid=True)
Z = gam.partial_dependence(term=0, X=XX, meshgrid=True)
ax.plot_surface(XX[1], XX[0], Z, cmap='viridis')
ax.azim = -45

# 2D contour
ax = fig.add_subplot(1, 2, 2)
ax.set_xlabel("Velocity (km/s)")
ax.set_ylabel("Backfill Pressure (kPa)")
contour = ax.contourf(XX[1], XX[0], Z, 10)
ax.scatter(X[:, 0], X[:, 1])
fig.tight_layout()
fig.subplots_adjust(bottom=0.2)

cbar_ax = fig.add_axes([0.055, 0.05, 0.885, 0.04])
cb = fig.colorbar(contour, cax=cbar_ax, label="Change in Degree of Separation (mm)", orientation='horizontal', drawedges=False)
plt.show()



'''non tensor GAM'''
gam2 = LinearGAM(s(0) + s(1) + f(2))
gam2.gridsearch(X, y)
gam2.summary()


plt.figure();
fig, axs = plt.subplots(1,3);
titles = ['velocity', 'pressure', 'sabot']
for i, ax in enumerate(axs):
    XX = gam2.generate_X_grid(term=i)
    ax.plot(XX[:, i], gam2.partial_dependence(term=i, X=XX))
    ax.plot(XX[:, i], gam2.partial_dependence(term=i, X=XX, width=.95)[1], c='r', ls='--')
    if i == 0:
        ax.set_ylim(-2.5,7.5)
        ax.set_ylabel('Change in Degree of Separation (mm)')
    ax.set_title(titles[i]);
    ax.set_xlabel(titles[i])

