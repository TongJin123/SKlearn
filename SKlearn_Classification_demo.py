import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()  # 数据哭有花的数据
iris_X = iris.data   # 取出花的所有属性
iris_y = iris.target  # 取出花的所有分类

print(iris_X[:2,:])  # 每个花都有四种属性，取了两个demo
print(iris_y)  # 花有三种类别

X_train,X_test,y_train,y_test = train_test_split(iris_X,iris_y,test_size=0.3)
# 使用train_test_split分开测试和学习的数据，这样会有一个好处，不会互相影响，不会因为人为因素造成误差
# test_size=0.3表示测试的数据占了总数据的30%，说明学习的数据占了70%,而且打乱了数据

knn = KNeighborsClassifier()  # 定义哪一个学习模块的方式
knn.fit(X_train,y_train)   # fit 把需要学习的数据放里面去，可以自己学习
print(knn.predict(X_test))  # knn是已经学习完的tnn,predict预测的那种花
print(y_test) # 真实的值