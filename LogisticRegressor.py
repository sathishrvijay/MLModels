"""
__Author__ : Vijay Sathish
__Date__	 : 08/16/2015

	- Logistic Regression
	Dimensions:
		X - mxn
		y - mx1
		theta - nx1

	Theory
		- With (Binary) Logistic Regression, y is either 0 (-ve class) or 1 (+ve class)
		- Hence we select a sigmoid function as the predictor yh
		- yh = g(theta * x) where
			- g(z) = 1/(1 - e^-z)
			- g'(z) = g(z) * (1 - g(z))				// Can be derived from basic calculus

	Maximum Likelihood
		- P(y = 1 | x, theta) = yh
		- P(y = 0 | x, theta) = 1 - yh
		Or L(theta) = P(y | x, theta) = yh^y * (1 - yh)^(1-y) 
		It is easier to maximize log likelihoood
			l(theta) = log L(theta) =  y*log(yh) + (1-y)*log(1 - yh)

		On going through the derivative of l(theta) wrt to theta, we arrive at the exact same update rule as in Linear Regression
		** Both LR and LoR can be shown as special cases of Generalized Linear Model

	Formulae:
		- yh = g(THETAT*X)
		- J(theta) = cost function = 1/2m * Sumj (y - yh)^2 for all X
			- where m is numObs
		- Use gradient descent to update thetai with learning rate alpha
				** Note - +ve sign in update comes after differentiation
				thetaj = thetaj + alpha/m * Sumi ((y - yh) * xj)i - lambda/m * thetaj			for all j > 0  
																																									(where lambda = C for L2 norm; 0 for no regularization)
				theta0 = theta0 + alpha/m * Sumi (y - yh)i 
		- SGD is the same set of update equations except that a random sub-set of the observations or selected on each iteration for updating each coefficient

		Note on Lambda and C -
			- The regularization term is usually lambda in literature
				J(theta) = A(theta) + lambda * B(theta) 
			- However, with SVM and scikit learn, it is convention to use C instead where C = 1/lambda
				J(theta) = C * A(theta) + B(theta) 
				- In this case, smaller the value of C, the greater the regularization 
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
	def __init__(self, penalty = None, alpha = 0.001, update_method = 'SGD', lam = 10, tol = 0.0001) :
		self.penalty = penalty
		self.updateMethod = update_method				# Choice of GD or SGD
		if (penalty == 'l2') :
			self.lam = lam						# regularization strength aka lambda or 1/2C
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
		z = self.theta0 + np.dot(self.X, self.theta)			# m x 1
		# print ("z.shape: ", z.shape)
		y_pred = 1 / (1 + np.exp(-z))											# m x 1
		# print ("y_pred.shape: ", y_pred.shape)
		# sys.exit()
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
			
			z = self.theta0 + np.dot(X_sel, self.theta)						# m' x 1
			y_pred = 1 / (1 + np.exp(-z))													# m' x 1
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
		while (iteration < 100000) :
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
	
	## Return the probabilities as is
	def predict_proba (self, X_test) :
		self.XTestNumObs = X_test.shape[0]
		assert (X_test.shape[1] == self.numFeatures)

		z = self.theta0 + np.dot(X_test, self.theta)
		y_pred_proba = 1 / (1 + np.exp(-z))											# m x 1
		return y_pred_proba

	## Convert probabilities to [0, 1]
	def predict (self, X_test) :
		y_pred_proba = self.predict_proba(X_test)
		y_pred = [1 if x > 0.5 else 0 for x in y_pred_proba]
		return y_pred

	## Compute prediction accuracy
	def score (self, X_test, y_test) :
		## Compute prediction
		y_pred = self.predict(X_test)
		
		acc_count = 0.
		for idx in range(X_test.shape[0]) :
			if (y_test[idx] == y_pred[idx]) :
				acc_count += 1
		
		## Print predictions for first few observations
		for test_idx in range(10) :
			print ("Obs_%d, y_test: %d, y_pred: %d" %(test_idx, y_test[test_idx], y_pred[test_idx]))
		return (acc_count / X_test.shape[0])


if __name__ == '__main__':
	## Set random seed
	np.random.seed(1234)

	update_method = 'GD'					# Choice of 'GD' or 'SGD'
	penalty = 'l2'								# Choice of 'l2' or None
	lam = 1												# Lambda parameter for regularization
	alpha = 0.01

	## Init the training set
	num_obs = 10000
	x1 = np.random.uniform(-10, 10, num_obs)
	x2 = np.random.normal(0, 2, num_obs)
	x3 = np.random.normal(2, 10, num_obs) 
	x4 = np.random.chisquare(10, num_obs) 
	X = np.column_stack([x1, x2, x3, x4])
	print ("X.shape: ", X.shape)
	y = np.zeros((num_obs, 1))
	for idx in range(num_obs) :
		y[idx, 0] = np.sign(10 + x1[idx] + 4.2*x2[idx] - 3*x3[idx] + 1.5*math.log(x4[idx]))
	y = (y + 1) / 2				# To constrain y to [0, 1]
	print ("y[0...10]: ", y[0:10].T)
	
	## Init the test set
	num_test_obs = 1000
	x1 = np.random.uniform(-8, 8, num_test_obs) 
	x2 = np.random.normal(0.5, 2.5, num_test_obs)
	x3 = np.random.normal(2, 11, num_test_obs)
	x4 = np.random.chisquare(9, num_test_obs) 
	X_test = np.column_stack([x1, x2, x3, x4])
	print ("X_test.shape: ", X_test.shape)
	y_test = np.zeros((num_test_obs, 1))
	for idx in range(num_test_obs) :
		y_test[idx, 0] = np.sign(10 + x1[idx] + 4.2*x2[idx] - 3*x3[idx] + 1.5*math.log(x4[idx]))
	y_test = (y_test + 1) / 2				
	print ("y_test.shape: ", y_test.shape)
	print ("y_test[0...10]: ", y_test[0:10].T)

	## Standard scale the features prior to fitting
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	X_test = scaler.transform(X_test)

	clf = LinearRegressor(penalty, alpha = alpha, update_method = update_method, lam = lam)
	clf.fit(X, y)
	score = clf.score(X_test, y_test)
	print ("Prediction accuracy: %0.4f" %(score))

"""
Output: 
	with GD: 
		- with alpha = 0.003, tol = 0.0001, stop after max 100k iterations
			thetas:  [[ 2.31665547]] [[ 1.75087911  2.5363487  -9.05146279  0.19184066]]
			Accuracy = 1.0
		- with alpha = 0.01,  penalty = 'l2', lam = 1, tol = 0.0001, converged after 18384 iterations
			thetas:  [[ 1.14021218]] [[ 0.75746737  1.10235344 -3.97048028  0.09046726]]
			Accuracy = 0.9880

	with SGD:
		- w/ alpha = 0.003, penalty = None, tol = 0.0001, converged after 34006 iterations
			thetas:  [[ 1.95773213]] [[ 1.6080803   2.16610342 -7.63052552  0.28970405]]
			Accuracy = 0.99

		- w/ alpha = 0.003, penalty = 'l2', lam = 10, tol = 0.0001, converged after 37979 iterations
			thetas:  [[ 0.76936734]] [[ 0.2153575   0.18629886 -0.78985162  0.05833896]]
			Accuracy = 0.8090

		- w/ alpha = 0.003, penalty = 'l2', lam = 1, tol = 0.0001, converged after 17521 iterations
			thetas:  [[ 1.17456666]] [[ 0.67771888  0.83115148 -3.1754527   0.11186786]]
			Accuracy = 0.9630

		Notes: 
		- We can see how increasing lambda constrains theta to lower values, but in this case lowers accuracy
		- In general, regularization helps increase stability of the solution and prevents over-fitting 
		- In sklearn.linear_model.Ridge, alpha is used which is the same as lambda or 1/2C
"""
