import numpy as np
from MiniFramework.NeuralNet_4_0 import *
from MiniFramework.ActivationLayer import *
from MiniFramework.DataReader_2_0 import *
from MiniFramework.ClassificationLayer import *




def dictionary_to_vector(dic_params):
    keys=[]
    count = 0
    for key in forward_Param:
        new_vector = np.reshape(dic_params[key],(-1,1))
        keys = keys + [key] * new_vector.shape[0]
        if count == 0:
            theta = new_vector
        else:  # np.concatenate
            theta = np.concatenate((theta, new_vector), axis=0)
        count +=1
    return theta,keys


def gradients_to_vector(gradients):
    count = 0
    for key in backward_Param:
        new_vector = np.reshape(gradients[key], (-1, 1))
        if count == 0:
            d_theta = new_vector
        else:
            d_theta = np.concatenate((d_theta, new_vector), axis=0)
        count = count + 1
    return d_theta


def vector_to_dictionary(theta, layer_dims):
    dict_params = {}
    L = 4
    start = 0
    end = 0
    for l in range(1,L):
        end += layer_dims[l]*layer_dims[l-1]
        dict_params["W" + str(l)] = theta[start:end].reshape((layer_dims[l-1],layer_dims[l]))
        start = end
        end += layer_dims[l]*1
        dict_params["B" + str(l)] = theta[start:end].reshape((1,layer_dims[l]))
        start = end
    #end for
    return dict_params

def CalculateLoss(net, dict_Param, X, Y, count, layer):
    for i in range(len(layer)):
        net.get_layer(layer[i]).weights.W = dict_Param["W" + str(i + 1)]
        # print("w"+str(i),dict_Param["W" + str(i + 1)][0,0])
        net.get_layer(layer[i]).weights.B = dict_Param["B" + str(i + 1)]
        # print("b"+str(i),dict_Param["B" + str(i + 1)][0,0])
    net.forward(X)
    p = Y * np.log(net.output)
    Loss = -np.sum(p) / count
    return Loss





if __name__ == '__main__':


    num_input = 7
    L = 4
    num_hidden1 = 16
    num_hidden2 = 12
    num_output = 10
    max_epoch = 40
    batch_size = 128
    learning_rate = 0.1
    eta = 0.2
    eps = 0.01

    params = HyperParameters_4_0(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.Fitting,
        init_method=InitialMethod.Xavier,
        stopper=Stopper(StopCondition.StopDiff, 1e-7))

    net = NeuralNet_4_0(params, "SAFE")
    fc1 = FcLayer_1_0(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")
    r1 = ActivationLayer(Sigmoid())
    net.add_layer(r1, "r1")

    fc2 = FcLayer_1_0(num_hidden1, num_hidden2, params)
    net.add_layer(fc2, "fc2")
    r2 = ActivationLayer(Sigmoid())
    net.add_layer(r2, "r2")

    fc3 = FcLayer_1_0(num_hidden2, num_output, params)
    net.add_layer(fc3, "fc3")
    r3 = ActivationLayer(Sigmoid())
    net.add_layer(r3, "r3")




    softmax = ClassificationLayer(Softmax())
    net.add_layer(softmax, "softmax")


    fc = ["fc1","fc2","fc3"]

    forward_Param = []

    dict_Param = dict()
    for i in range(1,L):
        dict_Param["W"+str(i)] = net.get_layer(fc[i-1]).weights.W
        dict_Param["B"+str(i)] = net.get_layer(fc[i-1]).weights.B
        forward_Param.append("W"+str(i))
        forward_Param.append("B"+str(i))


    layer_dims = [num_input, num_hidden1, num_hidden2,num_output]
    n_example = 2
    x = np.random.randn(n_example, num_input)
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(1, -1)

    net.forward(x)
    net.backward(x, y)

    dict_Grads = dict()
    backward_Param = []
    for i in range(1,L):
        dict_Grads["dW"+str(i)] = net.get_layer(fc[i-1]).weights.dW
        dict_Grads["dB"+str(i)] = net.get_layer(fc[i-1]).weights.dB
        backward_Param.append("dW" + str(i))
        backward_Param.append("dB" + str(i))



    J_theta, keys = dictionary_to_vector(dict_Param)
    d_theta_real = gradients_to_vector(dict_Grads)

    n = J_theta.shape[0]
    J_plus = np.zeros((n, 1))
    J_minus = np.zeros((n, 1))
    d_theta_approx = np.zeros((n, 1))

    # for each of the all parameters in w,b array
    for i in range(n):
        J_theta_plus = np.copy(J_theta)
        J_theta_plus[i][0] = J_theta[i][0] + eps

        J_plus[i] = CalculateLoss(net, vector_to_dictionary(J_theta_plus, layer_dims), x, y, n_example,fc)

        J_theta_minus = np.copy(J_theta)
        J_theta_minus[i][0] = J_theta[i][0] - eps
        J_minus[i] = CalculateLoss(net, vector_to_dictionary(J_theta_minus, layer_dims), x, y, n_example,fc)

        d_theta_approx[i] = (J_plus[i] - J_minus[i]) / (2 * eps)
    # end for
    numerator = np.linalg.norm(d_theta_real - d_theta_approx)  ####np.linalg.norm 二范数
    denominator = np.linalg.norm(d_theta_approx) + np.linalg.norm(d_theta_real)
    difference = numerator / denominator
    print('diference ={}'.format(difference))
    if difference < 1e-7:
        print("NO mistake.")
    elif difference < 1e-4:
        print("Acceptable, but a little bit high.")
    elif difference < 1e-2:
        print("May has a mistake, you need check code!")
    else:
        print("HAS A MISTAKE!!!")