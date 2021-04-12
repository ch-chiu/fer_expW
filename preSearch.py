import genetic as g
from tensorflow.keras.layers import Input, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization, DepthwiseConv2D, Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.activations import relu, sigmoid, softmax, softplus, softsign, tanh, selu, elu, exponential
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl
import time
# modify setting here
alpha_list = [i*0.25 for i in range(1, 12+1)] #12 choices
depth_multiplier_list = [1,2,3] #3 choices
activation_list = [relu, sigmoid, softmax, softplus, softsign, tanh, selu, elu] #8 choices
bias_list = [False, True] # 2 choices
dropout_list = [i*0.05 for i in range(0, 11)] # 11
pooling_list = [None, GlobalAveragePooling2D(), GlobalMaxPooling2D()] # 3 choices
optimizer_list = [SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl] # 8 choices
kernel_regularizer_list =[None,l1_l2] # 2 choices
bias_regularizer_list =[None,l1_l2] # 2 choices
activation_regularizer_list =[None,l1_l2] # 2 choices
layer_list = [1,2,3,4,5,6,7,8,9,10] #10 choices

paramRange1 = [alpha_list, depth_multiplier_list, activation_list, bias_list, dropout_list, pooling_list, optimizer_list, kernel_regularizer_list, bias_regularizer_list, activation_regularizer_list,layer_list]
paramRange2 = []

numOfGen1d = 1
numOfGen2d = 4
numOfParam = 11
start = time.time()
# 1d genetic
for i in range(numOfParam):
    p = g.population(paramRange1[i], i)
    for j in range(1, numOfGen1d+1):
        print("#########################################################")
        print("#########################################################")
        print("#########################################################")
        print(str(j) + "EVOLUTION" + " FOR PARAM " + str(i))
        print("#########################################################")
        print("#########################################################")
        print("#########################################################")
        f = open("log.txt", "a")
        f.write(str(j) + "EVOLUTION" + " FOR PARAM " + str(i)+"\n")
        f.close()
        p = g.evolve(p,0,extra=1)
    paramRange2.append(g.getParamRange(p, i))


f = open("log.txt", "a")
f.write("ParamRange,"+str(paramRange2[0])+","+str(paramRange2[1])+","+str(paramRange2[2])+","+str(paramRange2[3])+","+str(paramRange2[4])+","+
    str(paramRange2[5])+","+str(paramRange2[6])+","+str(paramRange2[7])+","+str(paramRange2[8])+","+str(paramRange2[9])+","+str(paramRange2[10])+"\n")
f.close()

end = time.time()

f = open("log.txt", "a")
cost = end - start
f.write("Time Spent\n")
f.write(str(cost))
f.close()
