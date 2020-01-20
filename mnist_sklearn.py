import numpy as np
from os import listdir
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# 加载数据并归一化
def img2vector(filename):
    retmat = np.zeros([1024], int)
    fr = open(filename)
    lines = fr.readlines()
    for i in range(32):
        for j in range(32):
            retmat[i * 32 + j] = lines[i][j]
    return retmat

def read_dataset(path):
    filelist = listdir(path)
    numfiles = len(filelist)
    dataset = np.zeros([numfiles, 1024], int)
    hwlabels = np.zeros([numfiles, 10])
    for i in range(numfiles):
        filepath = filelist[i]
        digit = int(filepath.split('_')[0])
        hwlabels[i][digit] = 1.0
        dataset[i] = img2vector(path + '/' + filepath)
    return dataset, hwlabels

'''
# 神经网络实现
train_dataset, train_hwlabels = read_dataset('trainingDigits')

clf = MLPClassifier(hidden_layer_sizes=100,
                    activation='logistic',
                    solver='adam',
                    learning_rate_init=0.0001,
                    max_iter=2000)
                    
dataset, hwlabels = read_dataset('testDigits')

res = clf.predict(dataset)
error_num = 0
num = len(dataset)
for i range(num):
    if np.sum(res[i] == hwlabels[i]) < 10:
        error_num += 1
print('Total num:', num, 'Wrong num:', \
      error_num, 'WrongRate:', error_num / float(num))
'''

# KNN实现
knn = KNeighborsClassifier(n_neighbors=3,
                           algorithm='kd_tree',)
knn.fit(train_dataset, train_hwlabels)

dataset, hwlabels = read_dataset('testDigits')
res = knn.predict(dataset)  # 不加载训练集，直接预测测试集
error_num = np.sum(res != hwlabels)
num = len(dataset)
print('Total num:', num, 'Wrong num:', \
      error_num, 'WrongRate:', error_num / float(num))