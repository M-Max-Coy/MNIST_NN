import numpy as np

'''
Define Neural Network with k=len(beta)-1 layers

Parameters:
data - (1xn) array of input data
beta - bias parameters
omega - weight parameters

Returns: 
all_f[k] - Final output of model as (1xn) array
all_f - Array of post layer calculations
all_h - Activations of post layer calculations
'''
def model(data,beta,omega):
    k = len(beta) - 1

    all_f = [None] * (k+1)
    all_h = [None] * (k+1)

    all_h[0] = data 

    for layer in range(k):
        all_f[layer] = beta[layer] + np.matmul(omega[layer],all_h[layer])
        all_h[layer+1] = ReLU(all_f[layer])

    all_f[k] = beta[-1] + np.matmul(omega[-1],all_h[-1])

    return all_f[k],all_f,all_h

'''
Maps array to array of probabilities
'''
def softmax(x):
    return np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x)))

'''
Loss function for multiclass regression. Negative log likelihood of categorical distribution

Parameters:
data - (1xn) array of input data
beta - bias parameters
omega - weight parameters

Returns:
Scalar representing loss
'''
def multiclass_loss(data,data_output,beta,omega):
    y_pred = model(data,beta,omega)[0]

    # return -1*(np.log(softmax(y_pred)[data_output]))
    return -1*(y_pred[data_output]-np.log(np.sum(np.exp(y_pred-np.max(y_pred))))-np.max(y_pred))

'''
Derivative of loss function wrt the model

Returns:
gradient vector
'''
def multiclass_loss_derivative(f,y):
    n = f.shape[0]

    grad = np.zeros(n)

    for j in range(n):
        grad[j] = -1*(int(y==j)-(np.exp(f[j]-np.max(f))/np.sum(np.exp(f-np.max(f)))))

    return grad

'''
ReLU activation function
'''
def ReLU(preactivation):
  activation = preactivation.clip(0.0)
  return activation

'''
Derivative of ReLU activation function
'''
def ReLU_derivative(x):
    x = np.array(x)
    x[x>=0] = 1
    x[x<0] = 0
    return x