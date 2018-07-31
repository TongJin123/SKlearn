from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
loaded_data = datasets.load_boston()
# 下载一个数据load，生成一个数据库就是make
data_X = loaded_data.data
data_y = loaded_data.target

model  = LinearRegression()
# 定义一个模型，用这个model来学习
model.fit(data_X,data_y)
# model.fit 来学习，传入数据的属性和类型

print(model.predict(data_X[:4,:]))
print(data_y[:4])

X,y = datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=1)
# 创造一些数据提供给我们学习，n_samples样例，n_features一个属性，targets一个分类，noise数据的离散程度
plt.scatter(X,y)
plt.show()