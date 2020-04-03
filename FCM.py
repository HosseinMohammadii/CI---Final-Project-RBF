import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle
from matplotlib import pyplot as plt


def create_centers(data, cluster_size, dimension_size):
    centers = np.zeros((cluster_size, dimension_size))
    dataa = data.T
    dim_max = []
    dim_min = []
    for i in range(0, len(dataa)):
        dim_max.append(max(dataa[i]))
        dim_min.append(min(dataa[i]))
    for i in range(0, cluster_size):
        centers[i] = [random.uniform(dim_min[0], dim_max[0]), random.uniform(dim_min[1], dim_max[1])]
    centers = np.array(centers)
    # print('centers created')
    # print(centers)
    return centers


def create_membership(cluster_size, data_size):
    n = data_size
    U = np.random.random((cluster_size, n))
    for k in range(0, n):
        s = sum(U[:, k])
        for i in range(0, cluster_size):
            U[i][k] = U[i][k] / s
    return U


def cal_membership(data_, centers_, cluster__size, data__size, mp):
    c = cluster__size
    n = data__size
    D = np.zeros((c, n))
    # print(centers_)
    for k in range(0, n):
        for i in range(0, c):
            # print('--')
            # print(k)
            # print(i)

            D[i][k] = np.linalg.norm(data_[k] - centers_[i])

    distances = D[:]

    "update memberships"
    u = np.zeros((cluster__size, data__size))
    n = len(distances[0])
    c = len(distances)
    Up = np.zeros((c, n))
    h = 2 / (mp - 1)
    for k in range(0, n):
        for i in range(0, c):
            if distances[i][k] == 0:
                u[i][k] = 0
            else:
                s = sum((distances[i][k] / distances[j][k]) ** h for j in range(0, c) if distances[j][k] != 0)
                u[i][k] = 1 / s
    membership = u[:]
    return membership


def fcm(data, data_size, cluster_size, iteration_size, dimension_size, mp):

    membershipo = create_membership(cluster_size, data_size)
    # print('len mememememe')
    # print(len(membershipo))
    centers = create_centers(data, cluster_size, dimension_size)
    # print(centers)
    curr = 0

    while curr < iteration_size:
        curr = curr + 1
        # print(centers)
        # print('------------')
        "update centers"
        c = len(membershipo)
        # print(c)
        n = len(data)
        m = len(data[0])
        V = np.zeros((c, m))
        # print(V)
        # print('+++++++++++++')
        for i in range(0, c):
            A = sum((membershipo[i][k] ** mp) * data[k] for k in range(0, n))
            b = sum(membershipo[i][k] ** mp for k in range(0, n))
            v = A / b
            V[i] = v
        centers = V[:]
        # print(V)
        # print('==============')


        "update distances"
        c = len(centers)
        n = len(data)
        D = np.zeros((c, n))
        for k in range(0, n):
            for i in range(0, c):
                D[i][k] = np.linalg.norm(data[k] - centers[i])
        distances = D[:]

        "update memberships"
        U = membershipo[:]
        n = len(distances[0])
        c = len(distances)
        Up = np.zeros((c, n))
        h = 2 / (mp - 1)
        for k in range(0, n):
            for i in range(0, c):
                if distances[i][k] == 0:
                    U[i][k] = 0
                else:
                    s = sum((distances[i][k] / distances[j][k]) ** h for j in range(0, c) if distances[j][k] != 0)
                    U[i][k] = 1 / s
        membership = U[:]

    plot(membershipo, centers, data)

    print('fcm finished')
    # print(centers)
    return centers, membership


def plot(U, centers, data):
    U = U.T
    for uu in U:
        armx = np.argmax(uu)
        uu[armx] = 1
    U = np.rint(U.T)

    c = len(centers)
    for i in range(0, c):
        t = np.where(U[i] == 1.)
        classD = np.array(data) [t]
        plt.scatter(classD[:, 0], classD[:, 1], s=30)
    plt.scatter(centers[:, 0], centers[:, 1], s=30, c='b', marker='s')
    plt.show()
