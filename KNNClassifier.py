"""
__Author__ : Vijay Sathish
__Date__	 : 08/16/2015

- K Nearest Neighbors Classifier
"""

import numpy as np
import math
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import minkowski
from scipy.stats import linregress


class KNNRegressor:
	def __init__(self, k, metric = 'euclidean', weighting = 'equal', p = 3) :
		self.k = k
		self.metric = metric
		self.weighting = weighting
		self.minkowski_p = p			# p applies only if distance metric is 'minkowski'
		self.X = []
		self.y = []
		self.numObs = 0
		self.numFeatures = 0

		## To store information related to X_test
		self.XTestNumObs = 0
		self.XTestNeighborsId = []
		self.XTestNeighborsDist = []
		self.XTestNeighborsY = []				# y values of nearest neighbors

	#### BEGIN - INTERNAL HELPER METHODS ####
	def compute_neighbors (self, X_test) :
		## Figure out the power p for minkowski distance
		if (self.metric == 'manhattan') :
			self.minkowski_p = 1
		elif (self.metric == 'euclidean') :
			self.minkowski_p = 2
		elif (self.metric == 'minkowski') :
			pass
		else :
			self.minkowski_p = 2
			print ("WARNING: Unknown distance metric: %s specified! Reverting to euclidean" %(self.metric))

		## Compute distance of a test point to all train points
		distance_matrix = np.zeros([self.numObs])
		neighbor_id = np.zeros([self.numObs])
		for test_idx in range(self.XTestNumObs) :
			for n_idx in range(self.numObs) :
				neighbor_id[n_idx] = n_idx
				distance_matrix[n_idx] = minkowski(X_test[test_idx], self.X[n_idx], self.minkowski_p)

			## Sort and record distances and neighbors
			sorted_distances = sorted(zip(distance_matrix, neighbor_id, self.y))
			for k_idx in range(self.k) :
				self.XTestNeighborsDist[test_idx, k_idx], self.XTestNeighborsId[test_idx, k_idx], self.XTestNeighborsY[test_idx, k_idx] = sorted_distances[k_idx]


	def compute_weighted_distance (self) :
		y_pred = []
		weights = []

		## Compute weights per test observation
		for test_idx in range(self.XTestNumObs) :
			if (weighting == 'equal') :
				weights = np.ones([self.k])
			elif (weighting == 'inverse') :
				weights = 1. / self.XTestNeighborsDist[test_idx, :]
			elif (weighting == 'exponential') :
				weights = 1. / np.exp(self.XTestNeighborsDist[test_idx, :])
			else :
				print ("ERROR: Unknown weighting = %s specified!" %(weighting))
				sys.exit()

			## Computed weighted distance
			# print ("Weights: ", weights)
			# print (self.XTestNeighborsDist[test_idx, :])
			y_pred.append(np.average(self.XTestNeighborsY[test_idx, :], axis = 0, weights = weights))

		return y_pred

	#### END - INTERNAL HELPER METHODS ####

	def fit (self, X, y) :
		self.X = X
		self.y = y
		self.numObs = self.X.shape[0]
		self.numFeatures = self.X.shape[1]


	def predict_proba (self, X_test) :
		self.XTestNumObs = X_test.shape[0]
		self.XTestNeighborsId = np.zeros([self.XTestNumObs, self.k])
		self.XTestNeighborsDist = np.zeros([self.XTestNumObs, self.k])
		self.XTestNeighborsY = np.zeros([self.XTestNumObs, self.k])
		if (self.numFeatures != X_test.shape[1]) :
			print ("ERROR: Train features=%d != test features=%d" %(self.numFeatures, X_test.shape[1]))
			sys.exit()
		
		## Compute neighbors per test observation
		self.compute_neighbors(X_test)

		## Compute prediction based on neighbors
		y_pred_proba = self.compute_weighted_distance()

		return y_pred_proba
	
	## Convert probabilities to discrete labels [0, 1]
	def predict (self, X_test) :
		y_pred_proba = self.predict_proba (X_test)
		y_pred = [1 if x > 0.5 else 0 for x in y_pred_proba]
		return y_pred

	def plot_prediction (self, X_test, y_test) :
		pass

	## Compute r^2 for prediction
	def score (self, X_test, y_test) :
		## Compute prediction
		y_pred = self.predict(X_test)
		
		## Print neighbor ids and distances for first few observations
		for test_idx in range(10) :
			print ("Obs_%d, y_test: %.4f, y_pred: %.4f, neighbors: " %(test_idx, y_test[test_idx], y_pred[test_idx]), self.XTestNeighborsY[test_idx])
			print ("neighbor distances: ", self.XTestNeighborsDist[test_idx, :])
			print ("\n") 

		## Compute Accuracy
		acc_score = 0.
		for idx in range(X_test.shape[0]) :
			if (y_test[idx] == y_pred[idx]) :
				acc_score += 1
		return (acc_score / X_test.shape[0])

	def kneighbors (self, X_test, return_distance) :
		if (return_distance) :
			return self.XTestNeighborsId, self.XTestNeighborsDist
		else :
			return self.XTestNeighborsId


if __name__ == '__main__' :
	## Set random seed
	np.random.seed(12346)

	k = 8
	distance = 'minkowski'
	weighting = 'exponential'

	## Init the training set
	num_obs = 10000
	x1 = np.random.uniform(-10, 10, num_obs)
	x2 = np.random.normal(10, 2, num_obs)
	x3 = np.random.normal(100, 10, num_obs) 
	x4 = np.random.chisquare(10, num_obs) 
	X = np.column_stack([x1, x2, x3, x4])
	print ("X.shape: ", X.shape)
	y = np.zeros([num_obs, 1])
	for idx in range(num_obs) :
		y[idx] = np.sign(x1[idx] + x2[idx]**2 - x3[idx] + math.log(x4[idx]))
	y = (y + 1) / 2				# To constrain y to [0, 1]
	print ("y[0...10]: ", y[0:10].T)
	
	## Init the test set
	num_test_obs = 1000
	x1 = np.random.uniform(-8, 8, num_test_obs) 
	x2 = np.random.normal(9, 2.5, num_test_obs)
	x3 = np.random.normal(95, 11, num_test_obs)
	x4 = np.random.chisquare(9.5, num_test_obs) 
	X_test = np.column_stack([x1, x2, x3, x4])
	print ("X_test.shape: ", X_test.shape)
	y_test = np.zeros((num_test_obs, 1))
	for idx in range(num_test_obs) :
		y_test[idx] = np.sign(x1[idx] + x2[idx]**2 - x3[idx] + math.log(x4[idx]))
	y_test = (y_test + 1) / 2				# To constrain y to [0, 1]
	print ("y_test.shape: ", len(y_test))
	print ("y_test[0...10]: ", y_test[0:10].T)

	## Standard scale the features prior to fitting
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	X_test = scaler.transform(X_test)

	clf = KNNRegressor(k, distance, weighting)
	clf.fit(X, y)
	score = clf.score(X_test, y_test)
	print ("Prediction Accuracy: %0.4f" %(score))

	print ("Plotting predictions...")
	# clf.plot_predictions()

"""
Output:

 k = 5; distance = 'euclidean'; weighting = 'inverse'  --> Acc = 0.977
 k = 8; distance = 'minkowski' p = 3; weighting = 'exponential' --> Acc = 0.976
 k = 10; distance = 'euclidean'; weighting = 'inverse'  --> Acc = 0.980
"""
