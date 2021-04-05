import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
def Gaussian_Distribution(N=2, M=10000, m=0, sigma=1):
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
'''一元高斯概率分布图'''
_, Gaussian = Gaussian_Distribution(N=1, M=1000, sigma=0.1)
x = np.linspace(-1,1,1000)
# 计算一维高斯概率
y = Gaussian.pdf(x)
plt.plot(x, y)
plt.title('Normal Distribution: $\mu = 0, $sigma = 0.1')
plt.xlabel('x')
plt.ylabel('probability')
plt.show()
