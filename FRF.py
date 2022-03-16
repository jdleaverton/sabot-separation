# -*- coding: utf-8 -*-
"""
Author: James Leaverton
"""
import numpy as np
from matplotlib import pyplot as plt
from math import atan2

'''
parameters
'''
steps = 180 # of samples between 600 and 2400 rpm

eccentricity = 7 # in

m = 0.125 # lbf * s^2 / in
M_w = 100 # lbf * s^2 / in
M_p = 37.65 # lbf * s^2 / in
M_a = 10 # lbf * s^2 / in

C_p = 1.32 * 32.127 * 12 # lbf * s / in

K_p = 7692.31 * 32.127 * 12 # lbf / in
K_a = 558.321 * 32.127 * 12 # lbf / in

originalNaturalFrequency = 146.9 # rad / s
dampingRatio = 0.0125

'''
creating M, C, K matrices
'''

M = np.array([[(M_w + M_p + m), 0.],
             [0., M_a]], dtype=float)

C = np.array([[C_p, 0.],
              [0., 0.]], dtype=float)

K = np.array([[(K_p + K_a), -K_a],
              [-K_a, K_a]], dtype=float)


# creating rpms linspace for graphing
def convertRPM(RPM):
    return (RPM * 2 * np.pi) / 60

frequency = np.linspace(600, 2400, steps, endpoint=True)

# initialize matrices
originalResponse = np.zeros(steps, dtype=float)
originalPhase = np.zeros(steps, dtype=float)
amplitude_washer = np.zeros(steps, dtype=float)
phase_washer = np.zeros(steps, dtype=float)
amplitude_absorber = np.zeros(steps, dtype=float)
phase_absorber = np.zeros(steps, dtype=float)

# calculations

for i in range(len(frequency)):
    
    operatingFrequency = convertRPM(frequency[i])
    
    # calculate original response
    r = operatingFrequency / originalNaturalFrequency
    J_r = (r**2) / np.sqrt((1 - r**2)**2 + (2 * dampingRatio * r)**2)
    originalResponse[i] = J_r * eccentricity * (m / (M_p + M_w + m))
    originalPhase[i] = atan2((-2 * dampingRatio * r), (1 - r**2)) * (180 / np.pi)
    
    # calculate F0 matrix
    F0 = np.array([[(m * eccentricity * (operatingFrequency ** 2))],
                     [0.]], dtype=float)
    
    # calculate dynamic stiffness matrix
    dyn1 = 1j * operatingFrequency * C
    dyn2 = -(operatingFrequency **2) * M
    
    dynamicStiffness = np.add(K, dyn1)
    dynamicStiffness = np.add(dynamicStiffness, dyn2)
    
    # calculate XP, linalg.solve(A,B) is a more accurate way of doing inv(A) * B - see documentation
    X_p = np.linalg.solve(dynamicStiffness, F0)
    
    # separate real and imaginary parts of X_p
    X_p_real = np.real(X_p)
    X_p_imag = np.imag(X_p)
    
    # magnitude of X_p for washer and absorber
    X_p_washer = np.sqrt((X_p_real[0])**2 + (X_p_imag[0])**2)
    X_p_absorber = np.sqrt((X_p_real[1])**2 + (X_p_imag[1])**2)
    
    # phase shift for washer and absorber - atan2 is takes (Y,X) and calculates atan(Y/X); it takes into account signs of Y and X
    phase_shift_washer = atan2(X_p_imag[0], X_p_real[0])
    phase_shift_absorber = atan2(X_p_imag[1], X_p_real[1])
    
    # equations of motion for washer and absorber assuming t = 1
    X_r_washer = X_p_washer * np.cos((operatingFrequency + phase_shift_washer))
    X_r_absorber = X_p_absorber * np.cos((operatingFrequency + phase_shift_absorber))
    
    # append to arrays
    amplitude_washer[i] = X_p_washer
    amplitude_absorber[i] = X_p_absorber
    phase_washer[i] = phase_shift_washer * (180 / np.pi)
    phase_absorber[i] = phase_shift_absorber * (180 / np.pi)
  

# plotting
plt.figure(figsize = (7, 5), dpi = 300)
plt.subplot(111)
plt.title('Amplitude of Frequency Response')
plt.xlabel('Operation Frequency (rpm)')
plt.ylabel('Amplitude (in)')
plt.yscale('log')
plt.plot(frequency, originalResponse, label='Original')
plt.plot(frequency, amplitude_washer, label = 'Washer')
plt.plot(frequency, amplitude_absorber, label = 'Absorber')
plt.legend(loc = 'lower right')

plt.figure(figsize = (7, 5), dpi = 300)
plt.subplot(111)
plt.title('Phase Lag of Frequency Response')
plt.xlabel('Operation Frequency (rpm)')
plt.ylabel('Phase Lag (degrees)')

plt.plot(frequency, originalPhase, 'b-', label = 'Original')
plt.plot(frequency, phase_washer, 'y-', label = 'Washer')
plt.plot(frequency, phase_absorber, 'r--', label = 'Absorber')
plt.legend(loc = 'lower right')















