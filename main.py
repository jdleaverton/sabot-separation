# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 17:27:03 2020

@author: JD
"""
import numpy as np
from matplotlib import pyplot as plt

'''
input parameters

'''

'tank dimensions'

# Blast tank length (travel of sabot from muzzle to sabot stripper) (in m)
travelLength = 2.5

# sabot stripper is assumed to be hollow disk with inner and outer diameter

# sabot stripper inner diameter (in cm)
stripperIDcm = 5
# convert to m
stripperID = stripperIDcm / 100 

# sabot stipper inner diameter
stripperODcm = 17
# convert to m 
stripperOD = stripperODcm / 100
# sabot stripper thickness (in cm)

stripperThicknesscm = 5
# convert to m 
stripperThickness = stripperThicknesscm / 100

'sabot dimensions'

# length of sabot (in mm)
sabotLengthmm = 25
# convert to cm
sabotLengthcm = sabotLengthmm / 10
# convert tom
sabotLength = sabotLengthmm / 1000

# width of sabot (in mm)
sabotWidthmm = 10
# convert to cm
sabotWidthcm = sabotWidthmm / 10
# convert to m
sabotWidth = sabotWidthmm / 1000

'main parameters for shot'

# integration steps
steps = 100

# Working environment gas
environmentGas = 'nitrogen'
adiabaticGasConstant = 1.4 # unitless, typically 1.4 for any diatomic gas. Air is primarily diatomic so it is assumed to be 1.4.
molecularWeight = 0.028014 # kg / mol
gasConstant = 8.31446261815324 # J / mol K

# Backfill Pressure of target/blast tank (in torr)
backfillPressure = 200
# Backfill Pressure in Pa (N / m^2)
backfillPressurePa = backfillPressure * 133.3223684211

# Temperature (in Celcius)
temperature = 21.5
temperatureKelvin = temperature + 273.15

# Expected Projectile Velocity (in m/s)
projVelocity = 3000

'''
calculations
'''

'initial calculations'


# sound speed of undisturbed air
soundSpeed = np.sqrt((adiabaticGasConstant * gasConstant * temperatureKelvin) / molecularWeight) # m / s

# Mach Number of projectile
machNumber = projVelocity / soundSpeed

gasConstantDensity = 0.082057366080960 # L * atm / mol * K
freestreamDensity = (1000 * molecularWeight * (backfillPressure / 760)) / (gasConstantDensity * temperatureKelvin) # in g / L




shockwavePressure = backfillPressurePa * (1 + ((2 * adiabaticGasConstant) / (adiabaticGasConstant + 1)) * ((projVelocity**2 / soundSpeed**2) - 1))


# projectile travel time between muzzle and sabot stripper in seconds
projTravelTime = (travelLength + stripperThickness) / projVelocity



'''
output graphs

'''

def createGraphs():
    
    projX = np.zeros(steps, dtype=float)
    projY = np.zeros(steps, dtype=float)
    projTimeInterval = projTravelTime / steps
    
    for i in range(steps - 1):
        projX[i+1] = projX[i] + projVelocity * projTimeInterval
    
    # graph's height
    # graphHeight = travelLength / 2.5
    
    
    
    # initialize figure
    plt.figure(figsize = (8.5, 11), dpi = 300)
    
    'create plot of projectile vs time'
    plt.subplot(211)
    plt.title('Sabot and Projectile Flight')
    plt.xlabel('X (in meters)')
    plt.ylabel('Zoomed Y (in meters)')
    
    # plt.axis([0, travelLength + stripperThickness, graphHeight / -2, graphHeight / 2])
    plt.axis([0, travelLength + stripperThickness, stripperOD / -2, stripperOD / 2])
    sabotStripperTop = plt.Rectangle((travelLength, stripperID / 2), stripperThickness, (stripperOD - stripperID) / 2, fc='grey', ec='black')
    sabotStripperBot = plt.Rectangle((travelLength, stripperID / -2), stripperThickness, (stripperOD - stripperID) / -2, fc='grey', ec='black')
    plt.gca().add_patch(sabotStripperTop)
    plt.gca().add_patch(sabotStripperBot)
    plt.plot(projX, projY, 'ko', markersize=1)
   
    
    'create figure of sabot stripper with impacts'
    plt.subplot(212)
    plt.title('Sabot Impact')
    plt.xlabel('X (in cm)')
    plt.ylabel('Y (in cm)')
    plt.grid(linewidth=0.75, color='black', alpha=0.2)

    yFinal = 5
    
    # create the sabot stripper on graph
    sabotStripperOD = plt.Circle((0,0), stripperODcm / 2, fc='grey', ec='black')
    sabotStripperID = plt.Circle((0,0), stripperIDcm / 2, fc='white', ec='black')
    plt.gca().add_patch(sabotStripperOD)
    plt.gca().add_patch(sabotStripperID)
    
    # graph the innermost point that a sabot piece could impact at
    innerBoundary = plt.Circle((0,0), yFinal, fill=False, linestyle='--')
    plt.gca().add_patch(innerBoundary)
    
    # add sabot impacts
    sabotTop = plt.Rectangle((-sabotWidthcm / 2, yFinal), sabotWidthcm, sabotLengthcm, fc='black')
    sabotRight = plt.Rectangle((yFinal, -sabotWidthcm / 2), sabotLengthcm, sabotWidthcm, fc='black')
    sabotBot = plt.Rectangle((-sabotWidthcm / 2, -(yFinal + sabotLengthcm)), sabotWidthcm, sabotLengthcm, fc='black')
    sabotLeft = plt.Rectangle((-yFinal, -sabotWidthcm / 2), -sabotLengthcm, sabotWidthcm, fc='black')
    plt.gca().add_patch(sabotTop)
    plt.gca().add_patch(sabotRight)
    plt.gca().add_patch(sabotBot)
    plt.gca().add_patch(sabotLeft)
    
    plt.axis('scaled')
    
    plt.show()
    

createGraphs()



