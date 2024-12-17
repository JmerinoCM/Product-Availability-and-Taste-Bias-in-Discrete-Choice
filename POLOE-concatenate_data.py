# -*- coding: utf-8 -*-
"""
POLOE: Probability of observing at least one exhaustion

Concatenate prob_n_noexh.csv to generate probs_n_nonexh_full.csv
"""

###############################################################################
## Libraries###################################################################
###############################################################################


import os
import pandas as pd

###############################################################################
## Setup ######################################################################
###############################################################################

directorio_principal = "C:/Projects/ProductAvaility/Repository"

###############################################################################
## Concatenate dfs ############################################################
###############################################################################

archivos_csv = []

for root, dirs, files in os.walk(directorio_principal):
    if 'prob_n_noexh' in root:
        for file in files:
            if file.endswith('.csv'):
                archivos_csv.append(os.path.join(root, file))

# Concatenate df and save information
dfs = []

for archivo in archivos_csv:
    df = pd.read_csv(archivo)
    if 'homog' in archivo:
        df['categoria'] = 'homog'
    elif 'heterog' in archivo:
        df['categoria'] = 'heterog'
    else:
        df['categoria'] = 'desconocido' 
    
    dfs.append(df)
df_concatenado = pd.concat(dfs, ignore_index=True)

ruta_guardado = os.path.join(directorio_principal, "probs_n_nonexh_full.csv")
if os.path.exists(ruta_guardado):
    os.remove(ruta_guardado)

df_concatenado.to_csv(ruta_guardado, index=False)
