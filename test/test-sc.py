from src import  kmeans,dealData

data = dealData.jk.tolist()
clus = kmeans.silhoustteCoefficient(data,maxK=20)
print(clus)


#测试欧氏距离是否计算正确
from src.similarityMeasure import euclideanDistance,cosine,cos_VW
import numpy as np
U = np.array([1,1,1,1])
V = np.array([-1,-1,-1,-1])
# print(euclideanDistance(U,V))
# 测试余弦距离是否正确
# print(cosine(U,V))
# print(cos_VW(U,V,1))