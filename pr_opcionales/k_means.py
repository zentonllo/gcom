# -*- coding: utf-8 -*-
"""
Created on Wed May 31 10:39:59 2017

@author: Alberto Terceño 
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def init_centroids(x,k):
    np.random.shuffle(x)
    return x[:k]

# Obtained from http://flothesof.github.io/k-means-numpy.html
def closest_centroids(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def adjust_centroids(x,centroids, k, assignments):
    return np.array([ x[ assignments == i, : ].mean(axis=0) for i in range(k)])
    
def kMeans(x, k, centroids, maxIters = 10):
        
    for i in range(maxIters):
        # Assign points to centroids
        assignments = closest_centroids(x, centroids)
     
        # Adjust centroids
        centroids = adjust_centroids(x, centroids, k, assignments)
        
    return centroids, assignments






# Modified from: https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot
class AnimatedKMeans(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, X, k, frames = 200, interval = 1000, step = 1):
        # Hacer assert de que X.shape[1] == 2
        self.numpoints = X.shape[0]
        self.X = X
        self.k = k        
        
        self.frames = frames
        self.interval = interval
        self.kmeans_step = step
        
        self.centroids = init_centroids(self.X, self.k)
                
        
        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=self.interval, 
                                           frames = self.frames, init_func=self.setup_plot, 
                                           blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        self.color = [0]*self.numpoints + [i for i in range(1, self.k + 1)]
        self.size = [1]*self.numpoints + [200]*self.k

        self.X_cent_stacked = np.vstack((self.X, self.centroids))        
        
        self.scat = self.ax.scatter(self.X_cent_stacked[:,0], self.X_cent_stacked[:,1], s=self.size, c=self.color, animated=True)      
        
        x_lower, x_upper = self.X[:,0].min()-1, self.X[:,0].max()+1
        y_lower, y_upper = self.X[:,1].min()-1, self.X[:,1].max()+1
        
        self.ax.axis([x_lower, x_upper, y_lower, y_upper])

        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,



    def update(self, i):
        """Update the scatter plot."""
        self.centroids, assignment = kMeans(self.X, self.k, self.centroids, maxIters = self.kmeans_step)
        
        self.color = [ j+1 for j in assignment.tolist()] + [j for j in range(1, self.k + 1)]        
        
        self.X_cent_stacked = np.vstack((self.X, self.centroids))
        # Set x and y data...
        #self.scat.set_offsets(self.X_cent_stacked)
        self.scat = self.ax.scatter(self.X_cent_stacked[:,0], self.X_cent_stacked[:,1], s=self.size, c=self.color, animated=True)        
        # Set sizes...
        #self.scat._sizes = 300 * abs(data[2])**1.5 + 100
        # Set colors..
        self.scat.set_array(np.array(self.color))
        
        

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def show(self):
        plt.show()

if __name__ == '__main__':
	m1, cov1 = [9, 8], [[1.5, 2], [1, 2]]
	m2, cov2 = [5, 13], [[2.5, -1.5], [-1.5, 1.5]]
	m3, cov3 = [3, 7], [[0.25, 0.5], [-0.1, 0.5]]
	data1 = np.random.multivariate_normal(m1, cov1, 250)
	data2 = np.random.multivariate_normal(m2, cov2, 180)
	data3 = np.random.multivariate_normal(m3, cov3, 100)
	X = np.vstack((data1,np.vstack((data2,data3))))
	np.random.shuffle(X)
	a = AnimatedKMeans(X = X, k = 3)
	a.show()


"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class AnimatedScatter(object):
    #An animated scatter plot using matplotlib.animations.FuncAnimation
    def __init__(self, numpoints=50):
        self.numpoints = numpoints
        self.stream = self.data_stream()

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5, 
                                           init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        #Initial drawing of the scatter plot
        x, y, s, c = next(self.stream)
        self.scat = self.ax.scatter(x, y, c=c, s=s, animated=True)
        self.ax.axis([-10, 10, -10, 10])

        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self):
        #Generate a random walk (brownian motion). Data is scaled to produce
        #a soft "flickering" effect
        data = np.random.random((4, self.numpoints))
        xy = data[:2, :]
        s, c = data[2:, :]
        xy -= 0.5
        xy *= 10
        while True:
            xy += 0.03 * (np.random.random((2, self.numpoints)) - 0.5)
            s += 0.05 * (np.random.random(self.numpoints) - 0.5)
            c += 0.02 * (np.random.random(self.numpoints) - 0.5)
            yield data

    def update(self, i):
        #Update the scatter plot
        data = next(self.stream)

        # Set x and y data...
        self.scat.set_offsets(data[:2, :])
        # Set sizes...
        self.scat._sizes = 300 * abs(data[2])**1.5 + 100
        # Set colors..
        self.scat.set_array(data[3])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def show(self):
        plt.show()

if __name__ == '__main__':
    a = AnimatedScatter()
    a.show()




"""