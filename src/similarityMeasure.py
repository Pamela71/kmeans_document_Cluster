# -*-coding:utf-8-*-
from functools import partial, reduce
from numpy import mat,inner,sqrt
import scipy.spatial.distance as dist
import math

#欧氏距离
def euclideanDistance(U,V):
    """U和V都是Ndarray对象,返回值是两点之间的欧氏距离"""
    return sqrt(inner(U-V,U-V))

#余弦
def cosine(U,V):
    """U和V都是ndarray对象，返回值是两向量的余弦值"""
    return inner(U,V) / sqrt(inner(U,U)*inner(V,V))

def vector_add(v, w):
    """adds two vectors componentwise"""
    return [v_i + w_i for v_i, w_i in zip(v,w)]

def vector_sum(vectors):
    return reduce(vector_add, vectors)

def scalar_multiply(c, v):
    return [c * v_i for v_i in v]

#计算向量的均值
def vector_mean(vectors):
    """compute the vector whose i-th element is the mean of the
    i-th elements of the input vectors"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))


def dot(v,w):
    return sum([v_i * w_i for v_i,w_i in zip(v,w)])

#余弦相似度距离s
def cos_VW(v,w,n):
    return dot(v,w)/math.sqrt(dot(v,v)*dot(w,w))*n

#jaccard距离
def jaccard_distance(v,w):
    matV = mat([v,w])
    return dist.pdist(matV, 'jaccard')