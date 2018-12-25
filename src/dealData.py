# -*-coding:utf-8 -*-
import pymssql
import jieba
from scipy import linalg
from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from .sqlsever import MSSQL

#获取数据
def getData():
    pass

# 提取文本数据的特征
def tranformMatrix(dataset, word=False, stop_words=None, vocabulary=None):
    """把文本数据转化为频率向量"""
    if word:
        wordset = [" ".join([k for k in dat]) for dat in dataset]
    else:
        wordset = [" ".join([k.strip() for k in jieba.cut(dat)]) for dat in dataset]
    vc = CountVectorizer(wordset, lowercase=False, stop_words=stop_words, vocabulary=vocabulary)
    return vc, vc.fit_transform(wordset).toarray()


# 数据降维
def jiangwei(data,percision=0.5):
    """对数据进行降维,data:样本特征矩阵,array类型;percision:丢失率，最大丢失多少原始数据信息.
    返回值是降维后的特征矩阵,array类型"""
    def pca_k(U, percision=percision):
        if isinstance(U, csr_matrix):
            U = U.toarray()
        u, s, v = linalg.svd(U)
        total = sum(s)
        i = 0
        k = 0.0
        while k < (1 - percision):
            i = i + 1
            k = sum([s[j] for j in range(i)]) / total
        return i

    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=pca_k(data), n_iter=100)
    return svd.fit_transform(data)


sqse = MSSQL()
sql = "SELECT * FROM dbo.class"
result = sqse.ExecQuery(sql)
labels = list(re[1] for re in result)  # 标签集
dats = list(re[2] for re in result)  # 数据集

# 第二次数据处理，减少簇的数量---每个簇至少包含4个样本点
sqlsec = "SELECT d_id,COUNT(c_id) '总数' FROM class group by d_id HAVING COUNT(c_id) > 4"
samplepoint = list(k[0] for k in sqse.ExecQuery(sqlsec))
result1 = []  # 数据集
labels1 = []  # 标签集
for re in result:
    if re[1] in samplepoint and re[1] != 0:
        result1.append(re[2])
        labels1.append(re[1])

stop_words = ['价格', '斤', '销售', '出售', '【', '】', '销量', '了', '大量', '供应', '批发', '优质', '山东']
vc, mat = tranformMatrix(result1, stop_words=stop_words)
jk = jiangwei(mat)