"""
__Author__ : Vijay Sathish
__Date__	 : 08/14/2015

- Base model to compute K-means clustering for the given data and predict cluster labels for input data
"""

import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean

class KMeans:
	"""
	Attributes: 
		k: number of centroids
		iterations: maximum number of iterations for a single run
		tol: relative tolerance with regards to inertia to declare convergence
	"""

	def __init__(self, k, iterations, tol=0.001) :
		self.numClusters = k
		self.maxIter = iterations
		self.tolerance = tol
		self.X = []
		self.clusterCenters = []				# List of the coordinates of the clusters
		self.XClusterId = []						# Records the cluster id for each of the observations
		self.XNearestClusterId = []			# Records the cluster id for the next nearest cluster
		self.systemInertia = 0.					# Total inertia of the system

	#### BEGIN - INTERNAL HELPER METHODS ####
	## Generate cluster indices and initialize those observations as cluster centroids
	def init_cluster_centers (self) :
		## generate random cluster indices
		cluster_indices = []
		while len(cluster_indices) < self.numClusters :
			draw = np.random.random_integers(0, self.X.shape[0] - 1, 1)
			if draw[0] not in cluster_indices :
				cluster_indices.append(draw[0])
		print ("Cluster indices: ", cluster_indices)
		
		for idx in cluster_indices:
			self.clusterCenters.append(self.X[idx, :])
		self.clusterCenters = np.array(self.clusterCenters)


	## Perform one iteration of assigning each observation to a cluster
	def compute_cluster_assignment (self) :
		for obs_idx in range(self.X.shape[0]) :
			## Compute euclidean distances between the observation and every cluster center and pick the smallest distance
			indices = [i for i in range(self.numClusters)]
			distances = [euclidean(self.X[obs_idx, :], self.clusterCenters[c_idx, :]) for c_idx in indices]
			sorted_distances = sorted(zip(distances, indices))
			#print (sorted_distances)

			## Now assign the first cluster_id to that observation in sorted_distances
			distance, self.XClusterId[obs_idx] = sorted_distances[0]
			## Also record the next nearest cluster which will be useful for measuring the Silhouette score
			distance, self.XNearestClusterId[obs_idx] = sorted_distances[1]
			#print ("Shortest distance cluster id: ", self.XClusterId[obs_idx])


	## Re-compute cluster centroids based on cluster assignment
	def recompute_cluster_centroids (self) :
		for c_idx in range(self.numClusters) :
			## Find the subset of observations belonging to c_idx cluster
			poi = [obs_idx for obs_idx in range(self.X.shape[0]) if self.XClusterId[obs_idx] == c_idx]

			## Create observation subset belonging to c_idx cluster and compute the mean
			obs_subset = []
			for obs_idx in poi :
				obs_subset.append(self.X[obs_idx])
			obs_subset = np.array(obs_subset)
			# print ("obs_subset.shape:", obs_subset.shape)
			self.clusterCenters[c_idx, :] = np.mean(obs_subset, axis = 0)
			# print ("new cluster_centroid[%d]: " %(c_idx), self.clusterCenters[c_idx, :])


	## Check for convergence
	def check_for_convergence (self, iteration) :
		inertia = self.compute_system_inertia(iteration)
		improvement = self.systemInertia - inertia 
		self.systemInertia = inertia
		if (improvement < self.tolerance) :
			return True
		else :
			return False


	## Compute total inertia of system for stopping criteria
	def compute_system_inertia (self, iteration) :
		inertia = 0.
		for c_idx in range(self.numClusters) :
			## Find the subset of observations belonging to c_idx cluster
			poi = [obs_idx for obs_idx in range(self.X.shape[0]) if self.XClusterId[obs_idx] == c_idx]
			
			## Create observation subset belonging to c_idx cluster and compute the mean
			obs_subset = []
			for obs_idx in poi :
				obs_subset.append(self.X[obs_idx])
			obs_subset = np.array(obs_subset)
			# print ("obs_subset.shape:", obs_subset.shape)
		
			## Compute distances of all the observations in a cluster from its centroid
			distances = [euclidean(obs_subset[obs_idx, :], self.clusterCenters[c_idx, :]) for obs_idx in range(obs_subset.shape[0])]
			inertia += np.sum(distances)

		print ("Iteration: %d, current inertia: %.3f, improvement: %.3f" %(iteration, inertia, self.systemInertia - inertia))
		return inertia

	## Compute Silhouette Score
	def compute_silhouette_score (self) :
		"""
			Silhouette score is bounded between (-1, 1)
				score = (dist_b - dist_a) / max(dist_a, dist_b)
			A score close to 1 implies that clustering is perfect, while -ve scores show that observations have been assigned to the wrong clusters
		"""
		silhouette_score = 0.
		for obs_idx, c_idx in enumerate(self.XClusterId) :
			dist_a = euclidean(self.X[obs_idx, :], self.clusterCenters[c_idx, :])
			dist_b = euclidean(self.X[obs_idx, :], self.clusterCenters[self.XNearestClusterId[obs_idx], :])
			silhouette_score += (dist_b - dist_a) / max(dist_a, dist_b)
		silhouette_score = silhouette_score / self.X.shape[0]

		return silhouette_score

	## get final cluster summary
	def print_cluster_summary (self) :
		print ("\nFinal system inertia after convergence: %.3f" %(self.systemInertia))
		for c_idx in range(self.numClusters) :
			## Find the subset of observations belonging to c_idx cluster
			poi = [obs_idx for obs_idx in range(self.X.shape[0]) if self.XClusterId[obs_idx] == c_idx]
			print ("Num observations in cluster_%d: %d" %(c_idx, len(poi)))
		
		print ("\n\nCentroid coordinates:")
		for c_idx in range(self.numClusters) :
			print ("centroid_%d: " %(c_idx), self.clusterCenters[c_idx])

		print ("Avg. silhouette_score: %.2f" %(self.compute_silhouette_score()))

	#### END - INTERNAL HELPER METHODS ####

	def fit (self, X) :
		" Initialize 'k' points as cluster centers randomly and then perform clustering until convergence "
		self.X = X
		self.XClusterId = np.zeros([self.X.shape[0]])
		self.XNearestClusterId = np.zeros([self.X.shape[0]])
	
		## Generate cluster indices and initialize those observations as cluster centroids
		self.init_cluster_centers()
		print ("clusterCenters.shape: ", self.clusterCenters.shape)

		## Initialize system inertia based on random assignment
		self.compute_cluster_assignment()
		self.systemInertia = self.compute_system_inertia(-1)

		for iteration in range(self.maxIter) :
			## Compute clusters
			self.compute_cluster_assignment()

			## Recompute cluster centroids after assignment
			self.recompute_cluster_centroids()
			# print ("\n\n")

			## Check whether to stop iterations
			if (self.check_for_convergence(iteration)) :
				print ("Clustering converged for iter: %d" %(iteration))
				break

	def predict (self, X) :
		return self.XClusterId

	def fit_predict (self, X) :
		self.fit(X)
		self.predict(X)

	def fit_transform (self, X) :
		pass

if __name__ == '__main__':
	## Set random seed
	np.random.seed(12346)

	k = 60
	num_obs = 1000
	num_iterations = 200
	x1 = np.random.uniform(-10, 10, num_obs)
	x2 = np.random.normal(10, 2, num_obs)
	x3 = np.random.normal(100, 10, num_obs) 
	x4 = np.random.chisquare(10, num_obs) 
	x5 = np.random.gamma(4, 5, num_obs) 
	x6 = np.random.gamma(10, 9, num_obs)
	x7 = np.random.chisquare(100, num_obs) 

	X = np.column_stack([x1, x2, x3, x4, x5, x6, x7])
	print ("X.shape: ", X.shape)

	"""
	print ("First ten rows of X...")
	for i in range(X.shape[0]) :
		if i > 10:
			break
		print (X[i, :])
	"""
	
	## Standard scale the input prior to clustering
	scaler = StandardScaler()
	X = scaler.fit_transform(X)

	"""
	print ("First five rows of X post scaling...")
	for i in range(X.shape[0]) :
		if i > 5:
			break
		print (X[i, :])
	"""
	clusterer = KMeans(k, num_iterations)

	## Perform clustering
	clusterer.fit(X)

	## Get cluster labels
	labels = clusterer.predict(X)

	## Print final summary
	clusterer.print_cluster_summary()


