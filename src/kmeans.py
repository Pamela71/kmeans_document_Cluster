# -*- coding:utf-8 -*-
import math
import random
from collections import defaultdict
from functools import reduce

from numpy import mat, argmax, array, int32, linspace
from . import similarityMeasure


class KMeans:
    """performs k-means clustering"""

    def __init__(self, k):
        self.k = k  # number of clusters
        self.means = None  # means of clusters

    def classify(self, input):
        return max(range(self.k),
                   key=lambda i: similarityMeasure.cos_VW(input, self.means[i], 1))

    def train(self, inputs, chose_k="sample"):
        if chose_k is "sample":
            self.means = random.sample(inputs, self.k)
        else:
            self.means = maxdistance_k(inputs, self.k)
        assignments = None

        while True:
            # Find new assignments
            new_assignments = list(map(self.classify, inputs))

            # If no assignments have changed, we're done.
            if assignments == new_assignments:
                return

            # Otherwise keep the new assignments,
            assignments = new_assignments

            for i in range(self.k):
                i_points = [p for p, a in zip(inputs, assignments) if a == i]
                # avoid divide-by-zero if i_points is empty
                if i_points:
                    self.means[i] = similarityMeasure.vector_mean(i_points)


# canopy算法，求出适合的K值
def canopy(datalist, T1, T2):
    """datalist is data set, T1 > T2 """
    canopy_CenterPoint = [[]]
    j = 0
    dis_k = []
    canopy_CenterPoint[0].append(datalist.pop(0))
    while len(datalist) != 0:
        for enum in datalist:
            for i, center in enumerate(canopy_CenterPoint):
                dis = similarityMeasure.cos_VW(enum, center, 1)
                dis_k.append(dis)
                if dis <= T1:
                    canopy_CenterPoint[i].append(enum)
                if dis <= T2:
                    datalist.remove(enum)
                    break
        if len(datalist) == 0:
            break
        if j >= 1:
            canopy_CenterPoint.append([])
            canopy_CenterPoint[-1].append(datalist.pop(0))
            j = 0
        j = j + 1
    # print(dis_k)
    return canopy_CenterPoint


# SSE曲线,找出最佳的K值
def plot_squared_clustering_errors(inputs):
    def squared_clustering_errors(inputs, k):
        """finds the total squared error from k-means clustering the inputs"""
        clusterer = KMeans(k)
        clusterer.train(inputs, chose_k="jkkj")
        means = clusterer.means
        assignments = list(map(clusterer.classify, inputs))

        return sum(similarityMeasure.cos_VW(input, means[cluster], 1) for input, cluster in zip(inputs, assignments))

    # p = math.floor(math.sqrt(len(inputs)))
    ks = list(range(2, 19 + 1))
    errors = [squared_clustering_errors(inputs, k) for k in ks]
    return (ks, errors)


def dot(v, w):
    """v_1 * w_1 + ... + v_n * w_n"""
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def maxCox(data):
    """data 是SSE曲线(ks,errors)的list 返回值是最佳的k值和得分数(k,score)"""

    def KMeanscos(u, v):
        if isinstance(u, list) and isinstance(v, list):
            a = [u[0] - u[1], v[0] - v[1]]
            b = [u[2] - u[1], v[2] - v[1]]
            cos = dot(a, b) / math.sqrt(dot(a, a) * dot(b, b))
            return u[1], cos

    datas = [KMeanscos(data[0][j:j + 3], data[1][j:j + 3]) for j in list(range(len(data[0]) - 2))]
    return max(datas, key=lambda x: x[1])


# 平均轮廓系数
def sicome(inputs, k):
    """data 是样本集,k是簇的数目,返回值是该簇数下的平均轮廓系数"""
    cluster = KMeans(k)
    cluster.train(inputs, chose_k='maxdistance')
    assignments = list(map(cluster.classify, inputs))
    inputs_index = defaultdict(list)
    for da, asss in zip(inputs, assignments): inputs_index[asss].append(da)

    def distance_f(data, cluster_data):
        return sum(1.0 - similarityMeasure.cosine(array(cls), array(data)) for cls in cluster_data)

    def point_sico(data, i, inputs_index):
        toala = distance_f(data, inputs_index[i]) / len(inputs_index[i])
        totlb = min([distance_f(data, inda) / len(inda)
                     for j, inda in inputs_index.items() if i != j])
        return (totlb - toala) / max((totlb, toala))

    return [sum(point_sico(data, i, inputs_index)
                for data, i in zip(inputs, assignments)) / len(assignments), k]


# 平均轮廓系数曲线确定最近加的K值
def silhoustteCoefficient(inputs, maxK='default'):
    """ inputs是样本集,maxK是最大的簇数目,默认的方法是取样本数的平方根的
    向下取整,不采用默认方法，可以输入任意一个整数值,返回值是最佳的k值和该k值下的轮廓系数值"""
    if maxK is 'default':
        maxK = math.floor(math.sqrt(len(inputs)))
    p = maxK
    return min([sicome(inputs, k)
                for k in range(2, int(p) + 1)], key=lambda x: x[0])


# 最大距离法选择簇类中心
def trainfor_matric(u):
    if isinstance(u, list):
        u = array(u)

    def dot(v, w):
        if isinstance(v, (int, int32)):
            return v * w
        return sum([v_i * w_i for v_i, w_i in zip(v, w)])

    return mat([[dot(j - i, j - i) for j in u] for i in u])


def maxdistance_k(u, k):
    U = trainfor_matric(u)
    raws, columns = U.shape

    def maxPosition(u):
        raw, column = u.shape
        _positon = argmax(u)
        return divmod(_positon, column)

    def f(u, v):
        return u * v

    k_point = []
    k_point.extend(maxPosition(U))
    for _ in range(k - 2):
        temp = [reduce(f, [U[j, i] for i in k_point]) for j in range(raws)]
        k_point.append(temp.index(max(temp)))
    return [u[ind] for ind in k_point]
