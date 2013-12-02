import numpy as np
# import scipy as sp
from scipy.special import erfinv
import matplotlib.pyplot as plt

N = 10000
y = np.random.rand(N)
x = np.sqrt(2) * erfinv(2*y - 1.0)
# y1 = np.ones(N)
# plt.figure(figsize=(8,6))
# plt.scatter(x,y1)
# plt.show()
# bins = np.linspace(-5,5,10)
# density, spc = np.histogram(x, bins=bins, density=True)
# print density,spc

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
# x = np.random.normal(0,1,1000)
numBins = 100
ax.hist(x,numBins,color='green',alpha=0.8)
plt.show()