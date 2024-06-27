from neural_network import model, multiclass_loss
from back_propagation import back_propagation

import numpy as np

'''
Find local minimum of function using adam + stochastic gradient descent
'''
def adam_stochastic_gradient_descent(data,data_output,bias,weights,batch_size,alpha=0.001,beta=0.9,gamma=0.999,MAX_ITERS=1000,epsilon=1e-6):
    k = len(bias) - 1

    n_data = data.shape[0]

    m_bias = [None] * (k+1)
    v_bias = [None] * (k+1)
    m_weights = [None] * (k+1)
    v_weights = [None] * (k+1)

    for i in range(k+1):
        m_bias[i] = np.zeros_like(bias[i])
        v_bias[i] = np.zeros_like(bias[i])
        m_weights[i] = np.zeros_like(weights[i])
        v_weights[i] = np.zeros_like(weights[i])

    iters_per_epoch = n_data//batch_size

    epoch_batch = 0 # Current iteration of epoch, resets every n_data/batch_size iters
    
    for j in range(MAX_ITERS):
        if j % 100 == 0:
            print(j)
            loss = 0
            for xo,yo in zip(data,data_output):
                loss += multiclass_loss(np.reshape(xo,(xo.shape[0],1)),yo,bias,weights)
            print(loss)

        if j % (iters_per_epoch*50) == 0:
            alpha *= 0.8

        # Choose random indices

        if j % iters_per_epoch == 0:
            shuffled_indices = np.random.permutation(np.arange(data.shape[0]))

        indices = shuffled_indices[batch_size*epoch_batch:batch_size*(epoch_batch+1)]
        batch_data = data[indices]
        batch_data_output = data_output[indices]

        # Find gradients of parameters wrt loss function
        dl_dbias = [None] * (k+1)
        dl_dweights = [None] * (k+1)

        for i in range(k+1):
            dl_dbias[i] = np.zeros_like(bias[i])
            dl_dweights[i] = np.zeros_like(weights[i])

        for observation,observation_output in zip(batch_data,batch_data_output):
            observation = np.reshape(observation,(observation.shape[0],1))
            _,cur_f,cur_h = model(observation,bias,weights)
            cur_dl_dbias,cur_dl_dweights = back_propagation(bias,weights,cur_f,cur_h,observation_output)
            for i in range(k+1):
                dl_dbias[i] += cur_dl_dbias[i]
                dl_dweights[i] += cur_dl_dweights[i]

        for i in range(k+1):
            dl_dbias[i] /= batch_size
            dl_dweights[i] /= batch_size
        
        # Adjust values of parameters
        for i in range(k+1):

            m_bias[i] = beta * m_bias[i] + (1-beta) * dl_dbias[i]
            v_bias[i] = gamma * v_bias[i] + (1-gamma) * dl_dbias[i] * dl_dbias[i]

            m_weights[i] = beta * m_weights[i] + (1-beta) * dl_dweights[i]
            v_weights[i] = gamma * v_weights[i] + (1-gamma) * dl_dweights[i] * dl_dweights[i]

            m_bias_tilde = m_bias[i] / (1-np.power(beta,i+1))
            v_bias_tilde = v_bias[i] / (1-np.power(gamma,i+1))

            m_weights_tilde = m_weights[i] / (1-np.power(beta,i+1))
            v_weights_tilde = v_weights[i] / (1-np.power(gamma,i+1))

            bias[i] = bias[i] - alpha * (m_bias_tilde/(np.sqrt(v_bias_tilde)+epsilon))
            weights[i] = weights[i] - alpha * (m_weights_tilde/(np.sqrt(v_weights_tilde)+epsilon))

    return bias,weights
