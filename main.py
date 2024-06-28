from csv_reader import read_csv
from neural_network import *
from optimize import *

import numpy as np

from PIL import Image

if __name__ == "__main__":
    # Training --------------------------------------------------
    train_file_name = "MNIST/mnist_train.csv"
    train_data = read_csv(train_file_name)

    D_I = 784
    D_1 = 16
    D_2 = 16
    D_O = 10

    np.random.seed(1)

    # He initialization of parameters
    beta0 = np.zeros((D_1,1))
    omega0 = 2/((D_I+D_1)/2) * np.random.randn(D_1,D_I)
    beta1 = np.zeros((D_2,1))
    omega1 = 2/((D_1+D_2)/2) * np.random.rand(D_2,D_1)
    beta2 = np.zeros((D_O,1))
    omega2 = 2/((D_2+D_O)/2) * np.random.randn(D_O,D_2)

    bias = [beta0,beta1,beta2]
    weights = [omega0,omega1,omega2]

    batch_size = 1000
    learning_rate = 0.0001
    beta = 0.9
    gamma = 0.999
    iters = 50000

    samples = 60000
    x = train_data[:samples,1:]
    for obv in x:
        mean = np.mean(obv)
        std = np.std(obv)
        obv = (obv-mean)/std
    y = train_data[:samples,0]
    print(f"x: {x.shape}, y: {y.shape}")
    bias_res,weight_res = adam_stochastic_gradient_descent(x,y,bias,weights,batch_size,learning_rate,beta,gamma,iters)

    for i in range(len(bias_res)):
        np.savetxt("parameters/bias_parameters" + str(i) + ".txt", bias_res[i], delimiter=","); np.savetxt("parameters/weight_parameters" + str(i) + ".txt", weight_res[i], delimiter=",")

    # Testing --------------------------------------------
    test_file_name = "MNIST/mnist_test.csv"
    test_data = read_csv(test_file_name)

    x_test = test_data[:,1:]
    for obv in x_test:
        mean = np.mean(obv)
        std = np.std(obv)
        obv = (obv-mean)/std
    y_test = test_data[:,0]

    count = 0
    for test_x,test_y in zip(x_test,y_test):
        res = softmax(model(np.reshape(test_x,(test_x.shape[0],1)),bias,weights)[0])
        if np.argmax(res) == test_y:
            count += 1

    print(count)
    print(count/len(y_test)*100)

    # Load saved parameters

    # beta0 = np.loadtxt("parameters/bias_parameters0.txt", delimiter = ",")
    # omega0 = np.loadtxt("parameters/weight_parameters0.txt", delimiter = ",")
    # beta1 = np.loadtxt("parameters/bias_parameters1.txt", delimiter = ",")
    # omega1 = np.loadtxt("parameters/weight_parameters1.txt", delimiter = ",")
    # beta2 = np.loadtxt("parameters/bias_parameters2.txt", delimiter = ",")
    # omega2 = np.loadtxt("parameters/weight_parameters2.txt", delimiter = ",")

    # bias = [beta0,beta1,beta2]
    # for i in range(len(bias)):
    #     bias[i] = np.reshape(bias[i],(bias[i].size,1))

    # weights = [omega0,omega1,omega2]

    # print(bias[0].shape)
    # print(omega0.shape)

    # # Test your own handwritten digits

    # file_name = "images/image.png"
    # image = Image.open(file_name)
    # gray_image = image.convert('L')
    # image_array = np.array(gray_image)
    # image_array = np.reshape(image_array,(784,1))
    # print(image_array.shape)

    # output = model(image_array,bias,weights)[0]
    # probabilities = softmax(output)
    # predicted_class = np.argmax(probabilities)
    # print(probabilities)
    # print(predicted_class)
