import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils import get_surrogate_model_coefficients


####################################################################
# x the intance that we want to explain
# z Perturbed simplified Imputs.  X_inter contains number_sample z
# x_prime = h(z).  X_sample contains number_sample x_prime
###################################################################

def create_interpretable_space(x):
    """
    Marks some each feature with the probability 2/len_feature
    I chosen 2/len_feature because with 2/len_feature there are a few 0
    """
    len_feature = x.shape[1]
    z = np.ones(len_feature)
    prob = 2/len_feature
    for i in range(len_feature):
        u = np.random.rand()
        if prob > u:
            z[i]= 0
    return z

def found_gaussien_feature_value(mu, std):
    return np.random.normal(mu, std)

def Kernel_distance(x, x_prime, kernel_width=0.75):
    """
    Computes the kernel distance
    """
    M = x.shape[0] 
    euclidean_dist = np.sqrt(np.sum((x - x_prime)**2))
    normalized_dist = euclidean_dist / (kernel_width * np.sqrt(M))
    weight = np.exp(-normalized_dist**2)
    return weight


def h_inter_to_origin(z, x, X):
    """
    Find a Gaussien approximation for element with Z = 0
    """
    x_prime = x.copy()
    for i in range(z.shape[-1]):
        if z[i] == 0:
            mu = X.iloc[:, i].mean()
            std = np.std(X.iloc[:,i])
            x_prime[0,i] = found_gaussien_feature_value(mu, std)
    return x_prime

def sampling_data_lime(h_inter_to_origin, x, X, number_sample):
    X_sample = np.zeros((number_sample, x.shape[1]))
    X_inter  = np.zeros((number_sample, x.shape[1]))
    i=0
    while i < number_sample:
        z = create_interpretable_space(x)
        if np.sum(z) != z.shape[-1]:
            x_prime = h_inter_to_origin(z, x, X)
            X_sample[i,:] = x_prime
            X_inter[i,:] = z
            i += 1
    return X_inter, X_sample

def My_LIMETabular(black_box_model, x, X, number_sample= 1000, kernel_width= 0.75, mode ="Regression", model="Ridge"):
    x = x.to_numpy()
    X_inter, X_sample = sampling_data_lime(h_inter_to_origin, x, X, number_sample)
    scaler = MinMaxScaler()
    scaler.fit(X)
    x = scaler.transform(x)  
    X_sample_scaled = scaler.transform(X_sample)
    weights = np.array([ Kernel_distance(x[0], X_sample_scaled[i], kernel_width) for i in range(len(X_sample)) ])
    Y = black_box_model.predict(X_sample)
    intercept, cofficients = get_surrogate_model_coefficients(X_inter, Y, weights , mode = mode, model = model, n_trials=50 )  
    return intercept, cofficients
