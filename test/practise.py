# a = [1,2,3,4]
# b = [1,2,3,4]
# k = []
# def f(m,b,x=k):
#     return x.append(m)
# j = list(map(f,a,b))
# print(k)
# print(j)

# from numpy import outer,array,inner,cross
#
# c1 = array([[1],[2],[3]])
# c2 = array([[1],[2],[3]])
# # print(type(c1))
# # c3 = outer(c1,c2)
# # print(c3)
# c4 = inner(c1,c2)
# print("inner %s" %c4)
# c5 = cross(c1.tolist(),c2.tolist())
# print("corss %s" %c5)


c = []
for i in range(10): c.append(i)
print(c)