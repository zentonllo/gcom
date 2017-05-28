"""

Main idea obtained from https://gist.github.com/bistaumanga/6023692

"""
import numpy as np

def init_centroids(x,k):
    np.random.shuffle(x)
    return x[:k]

#  The idea of building is taken from http://flothesof.github.io/k-means-numpy.html
def closest_centroids(x,centroids):
    distances = np.array([np.linalg.norm(x-c) for c in centroids])
    return np.argmin(distances)

def adjust_centroids(x,centroids, k, assignments):
    return np.array([ x[ assignments == i, : ].mean(axis=0) for i in range(k)])
    
    
def kMeans(x, k, maxIters = 10, plot_progress = None):

    centroids = init_centroids(X,k)
    for i in range(maxIters):
        # Cluster Assignment step
        assignments = closest_centroids(x, centroids)
     
        # Move centroids step
        centroids = adjust_centroids(x, centroids, k, assignments)
        #if plot_progress != None: plot_progress(X, C, np.array(centroids))
    return centroids, C


m1, cov1 = [9, 8], [[1.5, 2], [1, 2]]
m2, cov2 = [5, 13], [[2.5, -1.5], [-1.5, 1.5]]
m3, cov3 = [3, 7], [[0.25, 0.5], [-0.1, 0.5]]
data1 = np.random.multivariate_normal(m1, cov1, 250)
data2 = np.random.multivariate_normal(m2, cov2, 180)
data3 = np.random.multivariate_normal(m3, cov3, 100)
X = np.vstack((data1,np.vstack((data2,data3))))
np.random.shuffle(X)


centroids, C = kMeans(X, k = 3, plot_progress = None)