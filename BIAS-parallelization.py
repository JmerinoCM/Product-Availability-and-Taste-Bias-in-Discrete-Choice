#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BIAS: Bias of estimated betas as percentage of the true beta

Parallelization and automation of the run of the BIAS-code.py script for
different parameters of J, q, and degree of product differentiation
"""

###############################################################################
## Libraries###################################################################
###############################################################################

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor


###############################################################################
## Setup ######################################################################
###############################################################################

J_param = [100, 50, 10, 5] 
q_param = [
    [1 ,  2,  3,  5, 20, 50, 100],      
    [2 ,  3,  4,  5, 20, 50, 100], 
    [10, 11, 12, 15, 20, 50, 100],
    [20, 21, 22, 25, 40, 50, 100]
]


n_scripts = sum(len(sublist) for sublist in q_param) # Number of scripts to run simultaneously


simulacion_param = [1000, 100, 2024] # MC, n_consumers, seed

###############################################################################
## Functions ##################################################################
###############################################################################
     
def run_scripts(num_instancias, tipo_simula, valores_J, valores_q, valores_simulaciones):
    """
    Creates the folders for the execution of the script according to the
    combination of parameters, also executes the scripts in parallel
    """
    
    parametros = []
    carpetas_resultados = []
    
    if tipo_simula == "homog":
        val_uniforme = [0.75, 1.0]
    elif tipo_simula == "heterog":
        val_uniforme = [0.00, 1.0]
    else:
        val_uniforme = [] 
    
    for i in range(len(valores_J)):
        J = valores_J[i]
        sublista = valores_q[i]
        
        for param2 in sublista:

            parametros.append({"param1": J, "param2": param2,
                               "param_uni1": val_uniforme[0],
                               "param_uni2": val_uniforme[1],
                               "simu_p_1": valores_simulaciones[0],
                               "simu_p_2": valores_simulaciones[1],
                               "simu_p_3": valores_simulaciones[2]})

            carpeta = f"{tipo_simula}/J{J}"

            carpetas_resultados.append(carpeta)

    # Asegurarse de que las carpetas existen
    for carpeta in carpetas_resultados:
        os.makedirs(carpeta, exist_ok=True)

    # Función para ejecutar el script de simulación
    def run_py_file(params, carpeta):
        comando = f"python BIAS-code.py --param1 {params['param1']} --param2 {params['param2']} --param_uni1 {params['param_uni1']} --param_uni2 {params['param_uni2']} --simu_p_1 {params['simu_p_1']} --simu_p_2 {params['simu_p_2']} --simu_p_3 {params['simu_p_3']} --output {carpeta}"
        subprocess.run(comando, shell=True)

    # Ejecutar las simulaciones en paralelo
    with ThreadPoolExecutor(max_workers=num_instancias) as executor:
        for i, params in enumerate(parametros):
            carpeta = carpetas_resultados[i]
            executor.submit(run_py_file, params, carpeta)


###############################################################################
## Run functions ##############################################################
###############################################################################

run_scripts(n_scripts,
            "homog", 
            J_param,
            q_param,
            simulacion_param
            )

run_scripts(n_scripts,
            "heterog", 
            J_param,
            q_param,
            simulacion_param
            )

