#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 11:08:07 2025

@author: saul
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Definición de las ecuaciones del modelo SIR
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N         # Tasa de disminución de susceptibles
    dIdt = beta * S * I / N - gamma * I  # Tasa de cambio de infectados
    dRdt = gamma * I                 # Tasa de aumento de recuperados
    return dSdt, dIdt, dRdt

def main():
    # Parámetros del modelo
    N = 10000      # Población total
    I0 = 1        # Número inicial de infectados
    R0 = 0        # Número inicial de recuperados
    S0 = N - I0 - R0  # Número inicial de susceptibles

    beta = 0.3    # Tasa de transmisión (probabilidad de contagio por contacto)
    gamma = 0.1   # Tasa de recuperación

    # Intervalo de tiempo (en días)
    t = np.linspace(0, 160, 160)

    # Condiciones iniciales en un vector
    y0 = S0, I0, R0

    # Resolver las ecuaciones diferenciales con odeint
    sol = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = sol.T

    # Graficar el resultado
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptibles', linewidth=3)
    plt.plot(t, I, label='Infectados', linewidth=3)
    plt.plot(t, R, label='Recuperados', linewidth=3)
    plt.xlabel('tiempo (días)')
    plt.ylabel('número de personas')
    plt.title('simulación del modelo SIR')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
