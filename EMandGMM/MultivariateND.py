import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from jedi.api.refactoring import inline
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
##% matplotlib inline
def Gaussian_Distribution(N=2, M=1000, m=0, sigma=1):
    '''
    Parameters
    ----------
    N 维度
    M 样本数
    m 样本均值
    sigma: 样本方差

    Returns
    -------
    data  shape(M, N), M 个 N 维服从高斯分布的样本
    Gaussian  高斯分布概率密度函数
    '''
    mean = np.zeros(N) + m  # 均值矩阵，每个维度的均值都为 m
    cov = np.eye(N) * sigma  # 协方差矩阵，每个维度的方差都为 sigma

    # 产生 N 维高斯分布数据
    data = np.random.multivariate_normal(mean, cov, M)
    # N 维数据高斯分布概率密度函数
    Gaussian = multivariate_normal(mean=mean, cov=cov)

    return data, Gaussian
M = 100
data, Gaussian = Gaussian_Distribution(N=2, M=M, sigma=0.1)
# 生成二维网格平面
X, Y = np.meshgrid(np.linspace(-1,1,M), np.linspace(-1,1,M))
# 二维坐标数据
d = np.dstack([X,Y])
# 计算二维联合高斯概率
Z = Gaussian.pdf(d).reshape(M,M)


'''二元高斯概率分布图'''
fig = plt.figure(figsize=(6,4))
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='seismic', alpha=0.8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
