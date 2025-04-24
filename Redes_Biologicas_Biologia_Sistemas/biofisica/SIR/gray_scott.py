#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 16:02:37 2025

@author: saul
"""

import numpy as np
import matplotlib.pyplot as plt
#
# SIMULATION OF THE GRAY-SCOTT MODEL FOR REACTION-DIFFUSION
#
# References:
#
# This one explains the system and includes an online simulator:
# https://itp.uni-frankfurt.de/~gros/StudentProjects/Projects_2020/projekt_schulz_kaefer/
#
# Here there is a table of parameter values showing different behaviors:
# https://visualpde.com/nonlinear-physics/gray-scott.html
#
# -------------------------------
# Parameters for the simulation
# -------------------------------
F = 0.04       # Feed rate
k = 0.06       # Kill rate
Du = 0.16      # Diffusion rate of U
Dv = 0.08      # Diffusion rate of V

N = 200        # Grid size
T = 10000      # Total number of time steps
dt = 1.0       # Time step

# -------------------------------
# Initialize the concentration fields
# -------------------------------
U = np.ones((N, N))  # U is 1 everywhere initially
V = np.zeros((N, N)) # V is 0 everywhere initially

# Introduce small random noise into V in a central square region
r = 20
center = N // 2
# Python indexing: slice from center - r to center + r + 1 (inclusive equivalent)
V[center - r:center + r + 1, center - r:center + r + 1] = 0.25 + 0.01 * np.random.rand(2 * r + 1, 2 * r + 1)

# -------------------------------
# Define the Laplacian function using np.roll (periodic boundary conditions)
# -------------------------------
def laplacian(Z):
    return (np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
            np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) - 4 * Z)

# -------------------------------
# Set up the visualization
# -------------------------------
plt.figure(figsize=(6,6))
im = plt.imshow(V, cmap='plasma', vmin=0, vmax=0.6)  #
plt.colorbar()
plt.title('Gray-Scott Model (V concentration)')
plt.axis('equal')
plt.tight_layout()
plt.ion()  # Turn on interactive mode

# -------------------------------
# Main simulation loop
# -------------------------------
for t in range(1, T + 1):
    # Compute the reaction term
    reaction = U * V**2

    # Update the concentration fields
    U += dt * (Du * laplacian(U) - reaction + F * (1 - U))
    V += dt * (Dv * laplacian(V) + reaction - (F + k) * V)
    
    # Update the plot every 100 time steps
    if t % 1000 == 0:
        plt.figure(figsize=(6,6))
        im = plt.imshow(V, cmap='plasma', vmin=0, vmax=0.6)  #
        plt.colorbar()
        plt.title('Gray-Scott Model (V concentration)')
        plt.axis('equal')
        plt.tight_layout()
#        plt.ion()  # Turn on interactive mode


# -------------------------------
# Save the final pattern and display completion message
# -------------------------------
# Normalize V to [0, 1] for saving
final = V / np.max(V)
plt.imsave('gray_scott_final.png', final, cmap='plasma')
print('Simulation complete. Final pattern saved as "gray_scott_final.png".')

plt.ioff()
plt.show()
