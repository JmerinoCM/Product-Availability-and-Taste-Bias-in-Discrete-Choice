#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PONAE: Probability of a number of alternatives exhausted as the choice unfolds

Calculation of the probabilities that after each consumer chooses,
a certain number of products will be out of stock.
"""

###############################################################################
## Libraries###################################################################
###############################################################################

import numpy as np
import numba
import pandas as pd
from scipy.optimize import minimize
import warnings
import matplotlib.pyplot as plt
import os 
import shutil
from matplotlib.ticker import MaxNLocator
import argparse
import os

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
parser.add_argument('--foldername', type=str, required=True) 
parser.add_argument('--output', type=str, required=True) #outputh path

args_auto = parser.parse_args()

directorio_principal = args_auto.output
os.chdir(directorio_principal)
np.random.seed(seed=args_auto.simu_p_3)
MC =args_auto.simu_p_1
N_consumers = args_auto.simu_p_2
J = args_auto.param1
q_capacity = args_auto.param2
b_1 = 2
coeff_true = [b_1]
coeff_random = np.random.uniform(low=-10.0, high=10.0, size=len(coeff_true)).tolist() # Initial point for optimization
folder = args_auto.foldername 
X_1 = np.random.uniform(low  = args_auto.param_uni1,
                        high = args_auto.param_uni2,
                        size=J)

###############################################################################
## Functions ##################################################################
###############################################################################


## Functions for calculations for the nth consumer
@numba.njit
def utility_exp(coeff, feature_1, productos_unicos):
    
    u = coeff[0]*feature_1[productos_unicos] 
    return np.exp(u)

@numba.njit
def calculate_prob(coeff, feature_1, productos_unicos):

    utility_n = utility_exp(coeff, feature_1, productos_unicos)  
    den_n = np.sum(utility_n) 
    prob_n = utility_n / den_n 
    
    return prob_n.flatten()


## Functions for calculations of all consumers
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

@numba.njit
def tile_numba(arr, reps):
    """
    Implementing the numpy.tile function in numba
    """

    result = np.empty(len(arr) * reps, dtype=arr.dtype)
    for i in range(reps):
        result[i * len(arr):(i + 1) * len(arr)] = arr
    return result


@numba.njit
def list_seq(coeff_arg, feature_1, J_arg, R_arg, N_arg):
    
    """
    Creation of the choice dataset, takes into account when products
    are out of stock and consequently modifies consumers' choice sets
    """

    lista_productos = tile_numba(np.arange(0, J_arg), R_arg)

    productos_elegidos = np.empty(N_arg)
    
    for i_value in range(N_arg):
        
        productos_unicos =  np.unique(lista_productos)
        
        val_prob = calculate_prob(coeff_arg, feature_1, productos_unicos)
        
        list_ele = np.random.multinomial(1, val_prob, size = 1)
        
        indice = np.argmax(list_ele)
        

        productos_elegidos[i_value] = productos_unicos[indice]

        if productos_unicos[indice] in lista_productos:
            ee = np.where(lista_productos == productos_unicos[indice])[0][0]  # Encuentra el índice de la primera coincidencia
            lista_productos = np.delete(lista_productos, ee)

    return  productos_elegidos


def calculate_row_probabilities(arg_simulations, J, q_capacity, coeff_true, X_1, N_consumers):
    """
    Calculate the probabilities that a certain number of products in each row
    meet the condition m_choices_cumsum == q_capacity.
    """

    all_options = list(range(0, J))

    dataset_seq = list_seq(coeff_true, X_1, J, q_capacity, N_consumers)
    m_choices = pd.get_dummies(dataset_seq)
    m_choices = m_choices.reindex(columns=all_options, fill_value=False).to_numpy()
    num_rows = m_choices.shape[0]
    frequencies = np.zeros((num_rows, J + 1))

   
    for _ in range(arg_simulations):
       
        dataset_seq = list_seq(coeff_true, X_1, J, q_capacity, N_consumers)
        m_choices = pd.get_dummies(dataset_seq)
        m_choices = m_choices.reindex(columns=all_options, fill_value=False).to_numpy()
        m_choices_numeric = m_choices.astype(int)
        m_choices_cumsum = m_choices_numeric.cumsum(axis=0)
        count_columns_equal_q = (m_choices_cumsum == q_capacity).sum(axis=1)
        for row_index, count in enumerate(count_columns_equal_q):
            if count <= J:
                frequencies[row_index, count] += 1

    probabilities = frequencies / arg_simulations
    
    return probabilities

# Run functions

probabilities_n = calculate_row_probabilities(MC, J, q_capacity, coeff_true, X_1, N_consumers)


###############################################################################
## Save information ###########################################################
###############################################################################


carpeta_prob_noex = os.path.join(directorio_principal, folder)
os.makedirs(folder, exist_ok=True)

nombre_archivo = f"prob_n-J{J}_q{q_capacity}.csv"
ruta_archivo = os.path.join(folder, nombre_archivo)

if os.path.exists(ruta_archivo):
    os.remove(ruta_archivo)

df_probabilities = pd.DataFrame(probabilities_n, columns=[f'P_{k}' for k in range(J + 1)])

df_probabilities.to_csv(ruta_archivo, index=False)

