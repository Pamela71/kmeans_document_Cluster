import json
import math
import matplotlib.pyplot as plt

from .import dealData,kmeans
from sklearn import metrics
import numpy as np

dd = dealData.jk.tolist()
dd1 = dealData.mat.tolist()

def ARI(pre_true,pre_test):
    from sklearn import metrics
    return metrics.fowlkes_mallows_score(pre_true, pre_test)

def benchmark(pre_true,pre_test):
    H = []
    H.append(metrics.homogeneity_score(pre_true, pre_test))
    C = []
    C.append(metrics.completeness_score(pre_true, pre_test))
    V = []
    V.append(metrics.v_measure_score(pre_true, pre_test))
    A = []
    A.append(metrics.adjusted_rand_score(pre_true, pre_test))
    
def testARI(pre_true,data):
    ari = []
    for i in range(11):
        k = kmeans.KMeans(12)
        k.train(data,chose_k="jkskf")
        nowlabel= list(map(k.classify,data))
        ari.append(ARI(pre_true,nowlabel))
        # benchmark(pre_true,nowlabel)
    # with open('aris2-4.json','w') as f:
        # json.dump(ari,f)
    return ari

pca_test = testARI(dealData.labels1,dd)
test = testARI(dealData.labels1,dd1)

def test_plt(pca_test,test):
	x = list(range(len(test)))
	plt.plot(x,pca_test)
	plt.plot(x,test)
	plt.show()
test_plt(pca_test,test)
# dataerrors=kmeans.plot_squared_clustering_errors(dd)

# plt.plot(dataerrors[0], dataerrors[1])
# plt.xticks(dataerrors[0])
# plt.xlabel("k")
# plt.ylabel("total squared error")
# plt.show()
# import json
# with open('jjj.json','w') as fpp:
#     json.dump(dataerrors,fpp)