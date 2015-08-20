"""
__Author__ : Vijay Sathish
__Date__	 : 08/15/2015

	- Linear/Ridge Regression
	Dimensions:
		X - mxn
		y - mx1
		theta - nx1

	Formulae:
		- yh = theta0 + Sumi(thetai*xi) + lambda * Sumi(theta^2)
		- J(theta) = cost function = 1/2m * Sumj (y - yh)^2 for all X
			- where m is numObs
		- Use gradient descent to update thetai with learning rate alpha
				** Note - +ve sign in update comes after differentiation
				thetaj = thetaj + alpha/m * Sumi ((y - yh) * xj)i - lambda/m * thetaj			for all j > 0  
																																									(where lambda = C for L2 norm; 0 for no regularization)
				theta0 = theta0 + alpha/m * Sumj (y - yh)j 
		- SGD is the same set of update equations except that a random sub-set of the observations or selected on each iteration for updating each coefficient
"""

import numpy as np
import math
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import minkowski
from scipy.stats import linregress


class LinearRegressor:
	## L2 Regularization is implemented
	def __init__(self, penalty, alpha = 0.001, update_method = 'SGD', lam = 10, tol = 0.0001) :
		self.penalty = penalty
		self.updateMethod = update_method				# Choice of GD or SGD
		if (penalty == 'l2') :
			self.lam = lam						# regularization strength lambda aka 1/2C
		else :
			self.lam = 0
		self.alpha = alpha		# learning rate for Gradient Descent
		self.tol = tol				# tolerance level to calculate stoppage criterion
		self.X = []
		self.theta0 = np.random.randn(1, 1)
		self.theta = []				# Coefficients to be calculated
		self.y = []
		self.numObs = 0
		self.numFeatures = 0

		self.XTestNumObs = 0

	#### BEGIN - INTERNAL HELPER METHODS ####

	## Update theta using Gradient Descent
	def gd_update_coefficients (self) :
		y_pred = self.theta0 + np.dot(self.X, self.theta)			# m x 1
		# print ("y_pred.shape: ", y_pred.shape)
		theta0 = self.theta0 + (self.alpha / self.numObs) * np.sum(self.y - y_pred)			
		theta = self.theta + (self.alpha / self.numObs) * np.dot(self.X.T, (self.y - y_pred)) - (self.lam / self.numObs) * self.theta
		# print ("theta.shape: ", theta.shape)
		print ("thetas: ", theta0, theta.T)
		return theta0, theta
	
	## Update theta using Stochastic Gradient Descent
	# In SGD, only x% of the observations are randomly selected update theta (different for each theta)
	# Apart from being much faster than GD, it usually avoids local minima using this technique and improves convergence to global minima
	def sgd_update_coefficients (self) :
		num_update_points = int(self.numObs * 0.01)						# Use m' instead of m where m' = 0.01 * m
		X_sel = np.zeros([num_update_points, self.numFeatures])
		theta = np.zeros([self.numFeatures, 1])

		## Generate a random set of indices for each thetai and use those to update thetai
		for c_idx in range(-1, self.theta.shape[0]) :
			obs = np.random.uniform(0, self.numObs, num_update_points)
			for i, idx in enumerate(obs) :
				X_sel[i, :] = self.X[idx, :]
			y_sel = [self.y[idx] for idx in obs]
			
			y_pred = self.theta0 + np.dot(X_sel, self.theta)			# m' x 1
			# print ("y_pred.shape: ", y_pred.shape)
			if (c_idx == -1) :
				theta0 = self.theta0 + (self.alpha / num_update_points) * np.sum(y_sel - y_pred)			
			else :
				theta[c_idx] = self.theta[c_idx] + (self.alpha / num_update_points) * np.dot(X_sel.T, (y_sel - y_pred))[c_idx] - \
								(self.lam / self.numObs) * self.theta[c_idx]
		
		# print ("theta.shape: ", theta.shape)
		print ("thetas: ", theta0, theta.T)
		return theta0, theta
	
	#### END - INTERNAL HELPER METHODS ####


	def fit (self, X, y) :
		self.X = X																							# m x n
		self.numObs, self.numFeatures = X.shape
		self.theta = 10* np.random.randn(self.numFeatures, 1)				# n x 1
		print ("self.theta.shape: ", self.theta.shape)
		self.y = np.array(y)
		print ("self.y.shape: ", self.y.shape)
		
		## Repeat Gradient Descent until convergence
		iteration = 0
		while (True) :
			print ("Iteration_%d" %(iteration))
			if (self.updateMethod == 'GD') :
				theta0, theta = self.gd_update_coefficients()
			else : 			# Stochastic Gradient Descent
				theta0, theta = self.sgd_update_coefficients()

			iteration += 1
			# print ("theta.shape: ", theta.shape)
			diffs = np.abs(self.theta - theta)
			# print ("diff.shape: ", diff.shape)
			print ("diffs: ", diffs.T)
			
			## Update thetas
			self.theta0 = theta0
			self.theta = theta

			## Check for convergence
			for idx in diffs :
				if idx > self.tol :
					break
			else:
				if abs(theta0 - self.theta0)  < self.tol :
					break

	def predict (self, X_test) :
		self.XTestNumObs = X_test.shape[0]
		assert (X_test.shape[1] == self.numFeatures)

		y_pred = np.dot(X_test, self.theta)
		y_pred = self.theta0 + y_pred
		return y_pred

	## Compute r^2 for prediction
	def score (self, X_test, y_test) :
		## Compute prediction
		y_pred = self.predict(X_test)
		
		## Print neighbor ids and distances for first few observations
		for test_idx in range(10) :
			print ("Obs_%d, y_test: %.4f, y_pred: %.4f" %(test_idx, y_test[test_idx], y_pred[test_idx]))

		## Compute coefficient of determination
		slope, intercept, r_value, p_value, std_err = linregress(y_pred.T, y_test.T)
		return r_value**2


if __name__ == '__main__':
	## Set random seed
	np.random.seed(1234)

	update_method = 'SGD'					# Choice of GD or SGD
	penalty = 'l2'								# Choice of 'l2' or None
	lam = 10
	alpha = 0.0003

	## Init the training set
	num_obs = 10000
	x1 = np.random.uniform(-10, 10, num_obs)
	x2 = np.random.normal(10, 2, num_obs)
	x3 = np.random.normal(100, 10, num_obs) 
	x4 = np.random.chisquare(10, num_obs) 
	X = np.column_stack([x1, x2, x3, x4])
	print ("X.shape: ", X.shape)
	y = np.zeros((num_obs, 1))
	for idx in range(num_obs) :
		y[idx, 0] = 10 + x1[idx] + 4.2*x2[idx] - 3*x3[idx] + 1.5*math.log(x4[idx])
	
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
		y_test[idx, 0] = 10 + x1[idx] + 4.2*x2[idx] - 3*x3[idx] + 1.5*math.log(x4[idx])
	print ("y_test.shape: ", y_test.shape)

	## Standard scale the features prior to fitting
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	X_test = scaler.transform(X_test)

	regr = LinearRegressor(penalty, alpha = alpha, update_method = update_method, lam = lam)
	regr.fit(X, y)
	score = regr.score(X_test, y_test)
	print ("Prediction score (r^2): %0.4f" %(score))

"""
Output: 
	with GD: 
		- w/ alpha = 0.0003, penalty = None, toi = 0.00001, converged after 23404 iterations
			r^2 = 0.9999
	with SGD:
		- w/ alpha = 0.0003, penalty = None, toi = 0.00001, converged after 17280 iterations
			thetas:  [-243.00569674] [5.85474126   8.33775642 -29.71643592   0.7308375]
			r^2 = 0.9999
		- w/ alpha = 0.0003, penalty = 'l2', lam = 1, toi = 0.0001, converged after 11329 iterations
			thetas:  [[-236.17698968]] [[  4.4628479    6.17602413 -22.32832979   0.6828347 ]]
			r^2 = 0.9999
		- w/ alpha = 0.0003, penalty = 'l2', lam = 10, toi = 0.0001, converged after  iterations
			thetas:  [[-186.86791265]] [[ 1.39276568  1.81480716 -6.94230703  0.36452778]]
			r^2 = 0.9989
"""
