# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 23:01:17 2021

@author: JD
"""


import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap 
from mpl_toolkits.mplot3d import Axes3D

completedData = np.array([[6371.730044,	76,	    1.466854328],
                [1952.890831,	203,	1.817524762],
                [6536.846351,	76,   	1.37987809],
                [6295.467572,	75,   	1.502909766],
                [6252.668326,	75,   	1.389753122],
                [6484.108046,	100,	1.356073197],
                [2513.950161,	200,	1.946379685],
                [4054.476706,	100,	1.3531278],
                [1971.13232,	200,	1.709645283],
                [2033.78815,	200,	1.96598697],
                [2625.83491,	200,	1.77435836],
                [3233.763806,	200,	1.903090089],
                [3496.490089,	110,	1.50811434],
                [3106.913685,	200,	1.766520237],
                [3807.295031,	110,	1.610289218],
                [2264.2,	    300,	2.14336579],
                [5933.1,	    80,	    1.531408476],
                [3896.5,	    100,	1.629964538],
                [2434.9,	    299,	2.137230837],
                [2345.6,        199,    1.66176808],
                [2495.6,        215,    1.948865094],
                [5307.1,        100,    1.39665496],
                [4409.4,        100,    1.539907665],
                [1972.3,        197,    1.919583475],
                [2628.9,        200,    1.959092314],
                [6369.1,        85,     1.093763538],
                [2417.7,        199,    2.028162121],
                [3106.4,        102,    1.368782633],
                [3087.3,        100,    1.415385108],
                [5527.0,        90,     1.35479075],
                [6069.7,        90,     1.132869464],
                [2617.6,        149,    1.406661296],
],dtype=float)

futureData = np.array([[150, 4500],
                      [150, 3500],
                      [250, 2000],
                      [150, 2750],
                      [250, 3000],
                      [80, 5000],
                      [120, 5000]])

fig = plt.figure(figsize=[5.0, 3.0], dpi=300)
ax = fig.add_axes([0,0,1,1])
ax.scatter(completedData[:,1], completedData[:,0], label='Completed Tests')
ax.scatter(futureData[:,0], futureData[:,1], label='Future Tests')
ax.set_title('Degree of separation empirical model test matrix')
ax.set_xlabel('Backfill Pressure (Torr)')
ax.set_ylabel('Velocity (m/s)')
ax.legend()
ax.grid()
plt.show()

