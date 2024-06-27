import numpy as np
from neural_network import multiclass_loss_derivative,ReLU_derivative

'''
Find gradients of neural network parameters with respect to loss function
'''
def back_propagation(beta,omega,f,h,y):
    # Number of layers
    k = len(beta) - 1

    dl_dbeta = [None] * (k+1)
    dl_domega = [None] * (k+1)

    dl_df = [None] * (k+1)
    dl_dh = [None] * (k+1)

    dl_df[k] = np.array(multiclass_loss_derivative(f[k],y))

    for layer in range(k,-1,-1):
        dl_dbeta[layer] = np.reshape(dl_df[layer],(dl_df[layer].size,1))

        dl_domega[layer] = np.matmul(np.reshape(dl_df[layer],(dl_df[layer].size,1)),np.transpose(h[layer]))

        dl_dh[layer] = np.matmul(np.transpose(omega[layer]),np.reshape(dl_df[layer],(dl_df[layer].size,1)))
        
        if layer > 0:
            dl_df[layer-1] = ReLU_derivative(f[layer-1]) * dl_dh[layer]

    return dl_dbeta,dl_domega
