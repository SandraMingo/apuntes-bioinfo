#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 11:14:16 2025

@author: saul
"""
import numpy as np
import matplotlib.pyplot as plt

def gillespie_SIR(N, S0, I0, R0, beta, gamma, t_max):
    """
    Simula el modelo SIR utilizando el algoritmo de Gillespie.

    Parámetros:
      N     : Población total.
      S0    : Número inicial de susceptibles.
      I0    : Número inicial de infectados.
      R0    : Número inicial de recuperados.
      beta  : Tasa de transmisión (proporcional al número de contactos y probabilidad de infección).
      gamma : Tasa de recuperación.
      t_max : Tiempo máximo de la simulación.

    Retorna:
      times: Array con los instantes de tiempo en que se produce cada evento.
      S_list: Array con la evolución del número de susceptibles.
      I_list: Array con la evolución del número de infectados.
      R_list: Array con la evolución del número de recuperados.
    """
    # Inicialización
    t = 0.0
    S, I, R = S0, I0, R0
    times = [t]
    S_list = [S]
    I_list = [I]
    R_list = [R]
    
    while t < t_max and I > 0:  # Se simula hasta alcanzar t_max o no queden infectados
        # Cálculo de las tasas (propensiones)
        a_infection = beta * S * I / N
        a_recovery = gamma * I
        a_total = a_infection + a_recovery
        
        # Si no hay eventos posibles, se rompe el ciclo
        if a_total == 0:
            break
        
        # Tiempo hasta el siguiente evento (tau) basado en una distribución exponencial
        r1 = np.random.rand()
        tau = -np.log(r1) / a_total
        t += tau
        
        # Determinar qué evento ocurre usando una variable aleatoria
        r2 = np.random.rand()
        if r2 < a_infection / a_total:
            # Evento de infección: un susceptible se contagia
            S -= 1
            I += 1
        else:
            # Evento de recuperación: un infectado se recupera
            I -= 1
            R += 1
        
        # Almacenar los resultados en cada paso
        times.append(t)
        S_list.append(S)
        I_list.append(I)
        R_list.append(R)
    
    return np.array(times), np.array(S_list), np.array(I_list), np.array(R_list)

def main():
    # Parámetros de la simulación
    N = 10000      # Población total
    S0 = 9999      # Inicialmente, casi toda la población es susceptible
    I0 = 1        # Un caso infectado
    R0 = 0        # Ningún recuperado al inicio
    beta = 0.3    # Tasa de transmisión
    gamma = 0.1   # Tasa de recuperación
    t_max = 160   # Tiempo máximo de simulación (días)

    # Ejecutar la simulación
    times, S, I, R = gillespie_SIR(N, S0, I0, R0, beta, gamma, t_max)
    
    # Graficar los resultados utilizando gráficos en escalón (step plot)
    plt.figure(figsize=(10, 6))
    plt.step(times, S, where='post', label='Susceptibles', linewidth=2)
    plt.step(times, I, where='post', label='Infectados', linewidth=2)
    plt.step(times, R, where='post', label='Recuperados', linewidth=2)
    plt.xlabel("tiempo (días)")
    plt.ylabel("número de personas")
    plt.title("simulación estocástica del modelo SIR con el algoritmo de Gillespie")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
