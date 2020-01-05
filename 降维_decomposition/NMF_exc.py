import numpy as np
from sklearn.decomposition import NMF, PCA
from sklearn.datasets import fetch_olivetti_faces
from numpy.random import RandomState  # 创建随机种子
import matplotlib.pyplot as plt

# 加载数据的基本参数
n_row, n_col = 2, 3
n_components = n_row * n_col
image_shape = (64, 64)
dataset = fetch_olivetti_faces(shuffle=True, random_state=RandomState(0))  # 打乱顺序载入图像数据，并确定其种子
faces = dataset.data  # 加载数据的data内容
# 展示图片函数
def plot_gallery(title, images):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)

    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)  # i + 1表示每次激活哪个子图
        vmax = max(comp.max(), -comp.min())

        plt.imshow(comp.reshape(image_shape),
                   cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax,
                   vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)  # 图像之间的间隙

# 创建PCA与NMF模型
pca_model = PCA(n_components=6, whiten=True)
nmf_model = NMF(n_components=6, init='nndsvda', tol=5e-3)
estimators = [
    ('Eigenface - PCA using randomizes SVD', pca_model),
    ('Non-negative components - NMF', nmf_model)
]
# 训练模型
for name, estimator in estimators:
    estimator.fit(faces)  # 训练模型
    components_ = estimator.components_  # 获取提取的特征
    plot_gallery(name, components_[:n_components])

plt.show()
