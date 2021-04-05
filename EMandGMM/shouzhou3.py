import pandas as pd
import numpy as np
##%matplotlib inline
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
iris = datasets.load_iris()

df=pd.DataFrame(iris['data'])


kmeanModel = KMeans(n_clusters=3)
kmeanModel.fit(df)
df['k_means']=kmeanModel.predict(df)
df['target']=iris['target']

fig, axes = plt.subplots(1, 2, figsize=(8,6))
axes[0].scatter(df[0], df[1], c=df['target'],edgecolors='k')
axes[1].scatter(df[0], df[1], c=df['k_means'], cmap=plt.cm.Set1,edgecolors='k')
axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('K-Means', fontsize=18)

print(df.head())
iris['target']
distortions = []

K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df)
    distortions.append(kmeanModel.inertia_)
    plt.figure(figsize=(8,6))
plt.plot(K, distortions, 'bx-')
plt.xlabel('K')
plt.ylabel('SSE')
plt.title('The Elbow Algorithm showing the best K')
plt.show()

