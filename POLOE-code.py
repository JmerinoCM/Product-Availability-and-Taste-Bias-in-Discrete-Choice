#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POLOE: Probability of observing at least one exhaustion

Calculation of probabilities that at least one product is out of stock.
"""

###############################################################################
## Libraries###################################################################
###############################################################################

import numpy as np
import numba
import pandas as pd
import warnings
import os 
import shutil
import argparse
from scipy.stats import multinomial
import json

warnings.simplefilter("ignore", category=numba.NumbaWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
np.set_printoptions(precision=16) 

###############################################################################
## Setup ######################################################################
###############################################################################


parser = argparse.ArgumentParser()
parser.add_argument('--param1', type=int, required=True) # J
parser.add_argument('--param2', type=int, required=True) # q
parser.add_argument('--param_uni1', type=float, required=True) # a, Uniform distribution
parser.add_argument('--param_uni2', type=float, required=True) # b, Uniform distribution
parser.add_argument('--simu_p_1', type=int, required=True) #MC
parser.add_argument('--simu_p_2', type=int, required=True) #n_consumers
parser.add_argument('--simu_p_3', type=int, required=True) #seed
parser.add_argument('--output', type=str, required=True) #outputh path

args_auto = parser.parse_args()

directorio_principal = args_auto.output
os.chdir(directorio_principal)
np.random.seed(seed=args_auto.simu_p_3)
MC = args_auto.simu_p_1
N_consumers = args_auto.simu_p_2
J = args_auto.param1
q_capacity = args_auto.param2
b_1 = 2 # True coefficient
coeff_true = [b_1]
coeff_random = np.random.uniform(low=-10.0, high=10.0, size=len(coeff_true)).tolist() # Initial point for optimization
X_1 = np.random.uniform(low  = args_auto.param_uni1,
                        high = args_auto.param_uni2,
                        size=J)

###############################################################################
## Functions ##################################################################
###############################################################################


@numba.njit
def utility_exp_full(coeff, feature_1):
    
    u = coeff[0]*feature_1 
    return np.exp(u)

@numba.jit
def calculate_prob_full(coeff, feature_1):

    utility_n = utility_exp_full(coeff, feature_1)  
    den_n = np.sum(utility_n) 
    prob_n = utility_n / den_n 
    
    return prob_n.flatten()

def approximate_cdf(MC, probabilities, q, n):
    J = len(probabilities)  # Número de productos
    q_list = [q] * J
    q_minus_1_list = [item - 1 for item in q_list]
    
    simulations = multinomial.rvs(n, probabilities, size=MC)
    successes = np.all(simulations <= q_minus_1_list, axis=1)
    cdf_value = np.mean(successes)

    return cdf_value

# Run functions

probabilities_vals = list(calculate_prob_full(coeff_true, X_1))

cdf_values = []
for n in range(N_consumers, 0, -1):
    cdf_value = approximate_cdf(MC, probabilities_vals, q_capacity, n)
    cdf_values.append(cdf_value)

###############################################################################
## Save information ###########################################################
###############################################################################


folder = "prob_n_noexh"

carpeta_prob_noex = os.path.join(directorio_principal, folder)
os.makedirs(folder, exist_ok=True)

df = pd.DataFrame({
    "n_consumers": list(range(N_consumers, 0, -1)),  # consumer id
    "cdf_value": cdf_values, 
    "J": [J] * N_consumers,  
    "q_capacity": [q_capacity] * N_consumers, 
    "MC": [MC] * N_consumers,  
    "coeff_true": [coeff_true] * N_consumers, 
    "X_1": [X_1.tolist()] * N_consumers
})


nombre_archivo = f"J{J}_q{q_capacity}.csv"
ruta_archivo = os.path.join(folder, nombre_archivo)

if os.path.exists(ruta_archivo):
    os.remove(ruta_archivo)

df.to_csv(ruta_archivo, index=False)



