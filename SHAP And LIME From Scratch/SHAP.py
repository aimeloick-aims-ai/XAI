import numpy as np
import math
from utils import get_surrogate_model_coefficients
import itertools

####################################################################
# x the intance that we want to explain
# z Perturbed simplified Imputs.  X_inter contains 2^M z
# x_prime = h(z).  X_sample contains 2^M x_prime
###################################################################

def create_interpretable_space_shap(x):
    """
    Create the all coalition possible using nomber de feature
    So we have X_inter with (2^M, M) shape that is binary. N the number of feature
    """
    number_sample = x.shape[-1]
    all_combinations = list(itertools.product([0,1], repeat=number_sample))
    X_inter = np.array(all_combinations)
    return X_inter



def h_interpretable_to_origin(x, X, X_inter):
    """
    Transforme l'espace interprétable (binaire) en espace original pour une seule instance.
    x  The intance that we want to explain (1, M)
    X  Full DataFrame 
    X_inter    with shape (2^M, M)
    """
    # créer une copie en float pour X_sample
    X_sample = X_inter.copy().astype(float)

    # moyenne des colonnes sur tout X
    col_means = X.mean(axis=0).to_numpy()

    for j in range(X_inter.shape[0]):        
        for i in range(X_inter.shape[1]):    
            if X_inter[j, i] == 0:
                X_sample[j, i] = col_means[i]      
            else:
                X_sample[j, i] = x[0, i]
    return X_sample



def sampling_data_shap(x, X):
    """
    Use  last functions
    """
    X_inter  =  create_interpretable_space_shap(x)
    X_sample = h_interpretable_to_origin(x, X, X_inter)
    return X_inter, X_sample

def create_dict(X_inter, Y):
   """
   I finaly don't use now, I want to use after to comparaison
   """
   dict_all = {}
   for i in range(len(X_inter)):
      set_prov = tuple(X_inter[i,:])
      dict_all[set_prov] = Y[i]
   return dict_all
   

def SHAP_weight(X_inter, index):
    """
    Calculate the KernelSHAP weight with shapley vcoefficientfor the `index` row of X_inter 
    X_inter: binary  (2^N, M)
    index: row index
    """
    z = X_inter[index, :]
    M = z.shape[0]          
    s = np.sum(z)           

    if s == 0 or s == M:
        return 1e6         

    weight = (M - 1) / (math.comb(M, s) * s * (M - s))
    return weight
   

def My_SHAPTabular(black_box_model, x, X, mode ="Regression", model="Ridge"):
    """
    Main for SHAP
    Call all fonction and use
    """
    x = x.to_numpy()
    X_inter, X_sample = sampling_data_shap(x, X)
    Y = black_box_model.predict(X_sample)
    weights = np.array([ SHAP_weight(X_inter, i) for i  in range(len(X_inter)) ])
    intercept, cofficients = get_surrogate_model_coefficients(X_inter, Y, weights , mode = mode, model =model )  
    return intercept, cofficients
