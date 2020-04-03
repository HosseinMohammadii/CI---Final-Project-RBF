import math
import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import FCM


def create_rand_data(data, rand_number, dimension_size):
    rand_data = np.zeros((rand_number, dimension_size+1))
    dataa = data.T
    dim_max = []
    dim_min = []
    for i in range(0, len(dataa)):
        dim_max.append(max(dataa[i]))
        dim_min.append(min(dataa[i]))
    for i in range(0, rand_number):
        rand_data[i] = [random.uniform(dim_min[0], dim_max[0]), random.uniform(dim_min[1], dim_max[1]), 1.0]
    rand_data = np.array(rand_data)
    # print('centers created')
    # print(centers)
    return rand_data


def cal_G(data, centerss, memberShipp, gama, data__size):
    C = []
    cluster_size = len(centerss)
    for i in range(0, cluster_size):
        uu = 0.0
        sum_vec = np.zeros( (dimension_size, dimension_size) )

        for k in range(0, data__size):
            vecc = data[k] - centerss[i]
            vecc = vecc.T
            mat = vecc * vecc.T
            u = memberShipp[i][k]**m
            mat = u * mat
            sum_vec = sum_vec + mat
            uu = uu + u
        mat = 1/uu * sum_vec
        C.append(mat)

    print('C created')

    G_mat = [[0 for k in range(0, cluster_size)] for y in range(0, data__size)]

    gm = gama
    for k in range(0, data__size):
        for iii in range(0, cluster_size):
            cinverse = np.linalg.pinv(C[iii])
            v = data[k] - centerss[iii]
            v = np.array(v)
            vv = cinverse @ v.T
            vvv = v @ vv
            v2 = -1*gm*vvv
            G_mat[k][iii] = math.exp(v2)
    print('G created')
    return G_mat


def cal_w(G, class_data, cluster_sizeW, data_sizeW):
    g = G
    Y = np.zeros((data_sizeW, 2))

    for i in range(0, data_sizeW):
        if class_data[i][len(class_data[i]) - 1] == -1.0:
            Y[i] = [1, 0]
        if class_data[i][len(class_data[i]) - 1] == 1.0:
            Y[i] = [0, 1]
    print('created Y')

    g = np.array(g)
    gg = g.T @ g
    gg = np.array(gg)
    ggg = np.linalg.inv(gg)
    ggg = np.array(ggg)
    g2 = ggg @ g.T
    g2 = np.array(g2)
    W = g2 @ Y
    W = np.array(W)
    print('W created')
    return W


""" @:param train_data : train data with class column"""


def train(train_dataa, train_dataa_size, cluster_size, iteration_size, dimension_size, m, gama):

    """ remove class column"""
    train_data = train_dataa.T[:len(train_dataa.T) - 1].T

    centers, memberShip_train = FCM.fcm(train_data, train_dataa_size, cluster_size, iteration_size, dimension_size, m)

    """ to find out what G and W are see project definition"""
    G = cal_G(train_data, centers, memberShip_train, gama, train_dataa_size)
    W = cal_w(G, train_dataa, cluster_size, len(train_dataa))

    return W, centers


def test(test_dataa, test_dataa_size, cluster_size, m, centers, W, gama):
    test_data = test_dataa.T[:len(test_dataa.T) - 1].T
    membership_test = FCM.cal_membership(test_data, centers, cluster_size, test_dataa_size, m)
    G = cal_G(test_data, centers, membership_test, gama, test_dataa_size)
    G = np.array(G)
    yh = G @ W
    yhat = []
    for r in range(0, test_dataa_size):
        argmax = np.argmax(yh[r])
        yhat.append(argmax)
        # test_data[r].append(argmax)

    scatter(test_dataa)
    # scatter(test_data)
    plt.figure()

    f1 = f2 = 0
    for uo in range(0, len(test_data)):
        if yhat[uo] == 1:
            if f1 == 0:
                plt.scatter(test_data[uo][0], test_data[uo][1], color='red', label='Label=1')
                f1 = 1
            else:
                plt.scatter(test_data[uo][0], test_data[uo][1], color='red')
        else:
            if f2 == 0:
                plt.scatter(test_data[uo][0], test_data[uo][1], color='blue', label='Label=0')
                f2 = 1
            else:
                plt.scatter(test_data[uo][0], test_data[uo][1], color='blue')

    plt.title('Scattered data!')
    plt.legend(loc="upper left")
    plt.show()

    accuracy = 0
    for ty in range(0, test_dataa_size):
        if (test_dataa[ty][2] == 1.0 and yhat[ty] == 1.0) or (test_dataa[ty][2] == -1.0 and yhat[ty] == 0):
            accuracy = accuracy + 1
    accuracy = accuracy / test_dataa_size * 100
    print('accuracy is ', accuracy)


def scatter(data):
    plt.figure()

    f1 = f2 = 0
    for dat in data:
        if dat[2] == 1:
            if f1 == 0:
                plt.scatter(dat[0], dat[1], color='red', label='Label=1')
                f1 = 1
            else:
                plt.scatter(dat[0], dat[1], color='red')
        else:
            if f2 == 0:
                plt.scatter(dat[0], dat[1], color='blue', label='Label=0')
                f2 = 1
            else:
                plt.scatter(dat[0], dat[1], color='blue')

    plt.title('Scattered data!')
    plt.legend(loc="upper left")
    plt.show()


def cluster_border_plot(centers):
    rand_data_number = 1500
    rand_data = create_rand_data(dataa, rand_data_number, dimension_size)
    rand_data = rand_data.T[:len(rand_data.T) - 1].T
    membership = FCM.cal_membership(rand_data, centers, len(centers), len(rand_data), 2)
    FCM.plot(membership, centers, rand_data)

# program


if __name__ == '__main__':

    "read 2 dimension and two class data"
    fName = '2clstrain1200.csv'
    df = pd.read_csv(fName)
    X0 = []
    X1 = []
    Y0 = []
    X0.append(-1.58902338606647)
    for item in df['-1.58902338606647']:
        X0.append(item)
    X1.append(0.705860861189625)
    for item in df['0.705860861189625']:
        X1.append(item)
    Y0.append(1.0)
    for item in df['1.0']:
        Y0.append(item)

    "the pure data"
    dataa = np.vstack((X0, X1, Y0))

    "data without class"
    data = dataa[:len(dataa) - 1]
    data = data.T
    dataa = dataa.T
    dataa = np.array(dataa)
    data = np.array(data)

    "shuffle all pure data"
    shuffled_dataa = shuffle(dataa)

    "divide pure data into 70% and 30% for train and test"
    train_dataa = shuffled_dataa[: int(0.7 * len(shuffled_dataa))]
    test_dataa = shuffled_dataa[int(0.7 * len(shuffled_dataa)):]

    "plot all pure data with classes"
    scatter(dataa)

    gama = 1
    cluster_size = 3
    dimension_size = 2
    m = 2
    data_size = len(data)
    class_size = 2
    iteration_size = 70

    " train by train data"
    W, centers = train(train_dataa, len(train_dataa), cluster_size, iteration_size, dimension_size, m, gama)

    " to see clusters border"
    cluster_border_plot(centers)

    " test by train data"
    test(train_dataa, len(train_dataa), cluster_size, 2, centers, W, gama)

    " test by test data"
    test(test_dataa, len(test_dataa), cluster_size, 2, centers, W, gama)





