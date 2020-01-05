import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 加载数据
iris = load_iris()
x_train = iris.data
y_train = iris.target
# 构建模型并训练
pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
for i in range(len(x_train)):
    if y_train[i] == 0:
        red_x.append(x_train[i][0])
        red_y.append(x_train[i][1])
    elif y_train[i] == 1:
        blue_x.append(x_train[i][0])
        blue_y.append(x_train[i][1])
    else:
        green_x.append(x_train[i][0])
        green_y.append(x_train[i][0])
# 可视化表达
plt.scatter(red_x, red_y, c='r')
plt.scatter(blue_x, blue_y, c='b')
plt.scatter(green_x, green_y, c='g')
plt.show()