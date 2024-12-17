#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BIAS: Bias of estimated betas as percentage of the true beta

Creation of the dataset with capacity restriction, optimization of the standard
logit model and the one that takes into account the capacity restriction.
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

np.set_printoptions(precision=8) 

###############################################################################
## Setup ######################################################################
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--param1', type=int, required=True) # J
parser.add_argument('--param2', type=int, required=True) # q
parser.add_argument('--param_uni1', type=float, required=True) # a, Uniform distribution
parser.add_argument('--param_uni2', type=float, required=True) # b, Uniform distribution
parser.add_argument('--simu_p_1', type=int, required=True) #MC
parser.add_argument('--simu_p_2', type=int, required=True) # n_consumers
parser.add_argument('--simu_p_3', type=int, required=True) #seed
parser.add_argument('--output', type=str, required=True) # outputh path

args_auto = parser.parse_args()

directorio_principal = args_auto.output
os.chdir(directorio_principal)
method_opt = "l-bfgs-b" 
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

@numba.njit
def f_ll_noseq(coeff_arg, feature_1, matrix_choices_arg):
    
    """
    Standard logit model
    """
    m_probabilities = calculate_prob_full(coeff_arg, feature_1 )
    m_pji = matrix_choices_arg*m_probabilities
    m_pji_yji = np.sum(m_pji,axis=1)
    m_likelihood = -np.sum(np.log(m_pji_yji))
    
    return m_likelihood
    
@numba.njit
def f_ll_seq(coeff_arg, feature_1, dataset_seq_arg, J_arg, R_arg):
    
    """
    Capacity constraint logit model
    """
    
    N_consumers_arg = len(dataset_seq_arg)
    probs_productos_elegidos = np.empty(N_consumers_arg)
    stock_productos = tile_numba(np.arange(0, J_arg), R_arg)

    for i_value in range(N_consumers_arg):


        stock_productos_id_unicos = np.unique(stock_productos)
        val_prob = calculate_prob(coeff_arg, feature_1, stock_productos_id_unicos)
        id_j =  int(dataset_seq_arg[i_value])
        indice = np.where(stock_productos_id_unicos == id_j)[0][0]
        probs_productos_elegidos[i_value] = val_prob[indice]

        if id_j in stock_productos:
            ee = np.where(stock_productos == id_j)[0][0]  # Encuentra el índice de la primera coincidencia
            stock_productos = np.delete(stock_productos, ee)

        out_probs = probs_productos_elegidos
    
    m_likelihood = -np.sum(np.log(out_probs))
    
    return m_likelihood


###############################################################################
## Run optimizations ##########################################################
###############################################################################


columnas = ["model", "J", "q_capacity", "# consumers", "method",
            "success", "beta1",
            "fun_obj optimo", "fun_obj","secuencialidad de elecciones",
            "X_1"]
df_save = pd.DataFrame(columns=columnas)


for n_value in range(MC):
    
    # Dataset
    dataset_seq_save = list_seq(coeff_true, X_1, J, q_capacity, N_consumers)
    
    all_options_save = list(range(0, J))
    m_choices_save = pd.get_dummies(dataset_seq_save)
    m_choices_save = m_choices_save.reindex(columns=all_options_save, fill_value=False).to_numpy()

    # Optimizaciones
    
    ### Standard logit model
    res_min_save = minimize(f_ll_noseq,
                        coeff_random,
                        args=(X_1, m_choices_save),
                        method = method_opt,
                        options={'ftol': 1e-6})
    
    ff_nseq = f_ll_noseq(coeff_true, X_1, m_choices_save)
    
    ### Capacity constraint logit model
    
    res_min_seq_save = minimize(f_ll_seq,
                        coeff_random,
                        args=(X_1, dataset_seq_save, J, q_capacity),
                        method = method_opt,
                        options={'ftol': 1e-6})
    
    ff_seq = f_ll_seq(coeff_true, X_1, dataset_seq_save, J, q_capacity)
    
   # Save information to a dataframe

    df_save.loc[len(df_save)] = ["Sequential", J, q_capacity, N_consumers, method_opt,
                                 res_min_seq_save.success,
                                 res_min_seq_save.x[0],
                                 res_min_seq_save.fun,
                                 ff_seq,
                                 dataset_seq_save,X_1]
    
    df_save.loc[len(df_save)] = ["No Sequential", J, q_capacity, N_consumers,method_opt,
                                 res_min_save.success,
                                 res_min_save.x[0],
                                 res_min_save.fun,
                                 ff_nseq,
                                 dataset_seq_save,X_1]


###############################################################################
## Save information ###########################################################
###############################################################################



def guardar_df(df):
    
    """
    Save optimizations information in csv file
    """
    
    folder = 'dataframes'
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    save_path = os.path.join(folder, f"dataframe_Method-{method_opt}_J-{J}_q-{q_capacity}_N-{N_consumers}_MC-{MC}.csv")
    
    if os.path.exists(save_path):
        os.remove(save_path)
    
    df.to_csv(save_path, index=False)  # Guardar el DataFrame como CSV
    
    return save_path

def guardar_estadisticas(X_1, q=q_capacity):
    
    """
    Calculate the stats for X
    """

    promedio_X_1 = np.mean(X_1)
    varianza_X_1 = np.var(X_1)
    desviacion_estandar_X_1 = np.std(X_1)
    valor_minimo = np.min(X_1)
    valor_maximo = np.max(X_1)
    num_datos = np.size(X_1)

    data = {
        'Promedio': [promedio_X_1],
        'Varianza': [varianza_X_1],
        'Desviación Estándar': [desviacion_estandar_X_1],
        'Valor Mínimo': [valor_minimo],
        'Valor Máximo': [valor_maximo],
        'Num. datos': [num_datos]
    }
    df = pd.DataFrame(data)


    folder_path = os.path.join(os.getcwd(), 'feature')
    file_name = f"resultados_q_{q}.csv"
    file_path = os.path.join(folder_path, file_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if os.path.exists(file_path):
        os.remove(file_path)

    df.to_csv(file_path, index=False)

# Run save functions

guardar_df(df_save)
guardar_estadisticas(X_1)
