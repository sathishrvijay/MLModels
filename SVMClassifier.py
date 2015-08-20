"""
__Author__ : Vijay Sathish
__Date__	 : 08/19/2015

	Support Vector Classifier implementation (Binary Classification)
		- Includes soft margin support with L2 norm regularization
		- Kernel support

	Dimensions:
		X - mxn
		y - mx1
		alpha - mx1			--> Lagrange multipliers

	Theory
		- Let dot(w, x) + b = 0 represent the hyperplane that separates the positive and negative classes
			- We are interested in finding the hyperplane that maximizes the 'margin' aka the distance of the points that are closest to the separation hyperplane
			- These points closest to the decision boundary form the support vectors because they 'support' the decision boundary
		- Let the positive class be labelled +1 and negative as -1	
		- The decision function to classify a new data point x1 is now formulated as 
					D(x) = sign(dot(w, x1) + b) 		...1
					which is invariant to multiplying both w and b by some scalar lambda. 
		- Fixing the minimum distance of data points from decision boundary to 1, we get:
					yi * |dot(w, xi) + b| >= 1 				...2
					where the SVs have a distance of 1 from the separation hyperplane while others are farther away
		- Dividing ...2 by ||w|| on both sides to normalize w:
					yi * |dot(w, xi) + b| >= 1/||w|| 				...4
		- Maximizing ...4 is equivalent to minimizing 1/2 * w.w and adding lagrange multiplier alphai
					L(w, b) = 1/2 * w.w - Sumi {alphai * |yi* dot(w.xi + b) - 1|)} 		...5
		- Solving for this under the constraint that alphai >= 0 yields an equation that is quadratic in alpha (after expressing w and b in terms of alpha)
			- The parameters can be obtained by using quadratic optimization using cvxopt solver
					
	Soft Margins (L2 norm)
		- Further, if we want to avoid anomalies from affecting the decision boundary, we can introduce a slack term for soft margins as:
					yi * |dot(w, xi) + bi| >= 1 - epsi^2		...6

	Prediction 
		- Once alphai has been determined, the prediction is trivial 
					D(z) = sign(Sumj yj* alphaj* dot(xj, z) + b)		...7
					Note that the computation complexity is low because alphaj is 0 for all non support vector data points

	Kernels 
		- Some set of data points may not be linearly separable in the given dimension space but can be separated after transforming them into a higher dimension space 
		dot(xi, xj) --> <phi(xi), phi(xj)> --> K(xi, xj) where phi is mapping function into the higher dimension space
		- Since xj and z only appear in the dot or inner product, we can use a kernel (that satisifies Mercer's theorem) to replace the dot product with an arbitrary similarity function 
		- Additionally, the kernel trick helps us to achieve this WITHOUT having to figure out phi() or having to actually map the points into a higher dimensional space which would be extremely costly
 		- Note that dot(xj, z) can be replaced by K(xi, xj) in equation ...7 above with the kernel trick

"""

import numpy as np
import math
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean
# import cvxopt					## Quadratic Convex optimization solver

class Kernel :
	def __init__(self, kernel, gamma, degree, offset) :
		self.kernel = kernel
		self.gamma = gamma
		self.degree = degree
		self.offset = offset
	
	def returnKernelFunction (self) :
		if self.kernel == 'linear' :
			return self.linear
		elif self.kernel == 'rbf' or self.kernel == 'gaussian' :
			return self.rbf
		elif self.kernel == 'poly' :
			return self.poly
		else :
			print ("WARNING! Unknown kernel %s specified! Reverting to linear" %(self.kernel))
			return self.linear

	def linear (self, x1, x2) :
		return np.inner(x1, x2)
	
	def rbf (self, x1, x2) :
		exponent = -self.gamma * euclidean(x1, x2)
		return np.exp(exponent)

	def poly (self, x1, x2) :
		return (self.offset + np.inner(x1, x2)) ** self.degree


class SVMClassifier:
	## L2 Regularization is implemented
	def __init__(self, penalty = None, C = 1, kernel = 'linear', gamma = 0.1, degree = 3, offset = 0.) :
		self.penalty = penalty
		if (penalty == 'l2') :
			self.C = C						# 'l2' norm regularization strength
		else :
			self.C = 0
		
		## Setup the Kernel similarity function
		self.kernel = kernel		# Option to apply kernel trick for kernels that satisfy Mercer's theorem
		self.gamma = gamma			# radius of influence for Gaussian or Radial Basis Function kernel
		self.degree = degree  	# degree of polynomial for a polynomial kernel
		self.offset = offset		# bias or constant term for polynomial kernel
		kernel = Kernel (self.kernel, self.gamma, self.degree, self.offset)
		self.K = kernel.returnKernelFunction()

		self.numObs = 0
		self.numFeatures = 0

		## All of these will be learned during the fit process
		self.bias = 0.					# Bias coefficient b
		self.alpha = []					# Lagrange Coefficients for quadratic convex optimization
		self.SVX = []						# Support Vectors
		self.SVy = []						# Support Vector labels

		self.XTestNumObs = 0

	#### BEGIN - INTERNAL HELPER METHODS ####

	## Compute the Lagrange multipliers using quadratic convex optimization
	# Ref: https://gist.github.com/ajtulloch/7655467
	def computeMultipliers(self, X, y):
		K_matrix = self.returnSimilarityScoresMatrix(X)
		# Solves :
		# min 1/2 x^T P x + q^T x
		# such that
		#  Gx \coneleq h
		#  Ax = b

		P = cvxopt.matrix(np.outer(y, y) * K_matrix)
		q = cvxopt.matrix(-1 * np.ones(self.numObs))
		# -a_i \leq 0
		G_std = cvxopt.matrix(np.diag(np.ones(self.numObs) * -1))
		h_std = cvxopt.matrix(np.zeros(self.numObs))
	
		# a_i \leq c
		G_slack = cvxopt.matrix(np.diag(np.ones(self.numObs)))
		h_slack = cvxopt.matrix(np.ones(self.numObs) * self.C)
		G = cvxopt.matrix(np.vstack((G_std, G_slack)))
		h = cvxopt.matrix(np.vstack((h_std, h_slack)))
		A = cvxopt.matrix(y, (1, self.numObs))
		b = cvxopt.matrix(0.0)
		solution = cvxopt.solvers.qp(P, q, G, h, A, b)
	
		return np.ravel(solution['x'])

	## Returns matrix of f(inner_dot_product) depending on the kernel defined
	def returnSimilarityScoresMatrix (self, X) :
		K_matrix = np.zeros([self.numObs, self.numObs])
		for i, xi in enumerate(X) :
			for j, xj in enumerate(X) :
				# K_matrix[i, j] = np.inner(xi, xj)			# similar to np.dot() except that we don't need to deal with transpose in this case
				K_matrix[i, j] = self.K(xi, xj)
		return K_matrix

	## Extract support vectors and calculate bias
	def extractSupportVectors (self, X, y, lagrange_multipliers) :
		## Filter out only the alphas which are positive. These form the support vector indices
		sv_indices = [idx for idx in range(len(lagrange_multipliers)) if lagrange_multipliers[idx] > 0]

		self.alpha = lagrange_multipliers[sv_indices]
		self.SVX = []
		for idx in range(len(sv_indices)) :
			self.SVX.append(X[sv_indices[idx], :])
		slf.SVy = y[sv_indices]

		## Set the bias term
		# A straight-forward way to calculate this is to predict with bias of zero for support vectors and then commpute the error
		y_pred = []
		for xi, yi in zip(self.SVX, self.SVy) :
			y_pred = self.predict (self.SVX, no_bias = True)
		self.bias = np.mean(y - y_pred)
			
	
	#### END - INTERNAL HELPER METHODS ####

	def fit (self, X, y) :
		self.numObs, self.numFeatures = X.shape
		print ("y.shape: ", len(y))
		
		## Compute alphai and then set the support vectors
		lagrange_multipliers = self.computeMultipliers (X, y)
		self.extractSupportVectors (X, y, lagrange_multipliers)
		
	def predict (self, X_test, no_bias = False) :
		assert (X_test.shape[1] == self.numFeatures)
		
		y_pred = []
		## Compute the decision function D(z) = sign(Sumj yj* alphaj* dot(xj, z) + b) for each data point 'z' in the test set
		for x_test in X_test :
			result = self.bias
			if (no_bias == True) :					# This option is used to compute the bias itself
				result = 0.
			for alphai, xi, yi in zip(self.alpha, self.SVx, self.SVy) :
				# result += alphai * yi * np.inner(xi, x_test)
				result += alphai * yi * self.K(xi, x_test)
			y_pred.append(np.sign(result))
			
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

	kernel = 'linear'							# Choices of 'linear', 'rbf' or 'poly'
	degree = 3										# Degree of polynomial for 'poly' kernel
	gamma = 1											# Applicable to 'rbf' only: gamma controls radius of influence of a SV data point. Smaller values imply larger radius or smoother decision boundaries
	penalty = 'l2'								# Only L2 norm or regularizer is currently implemented
	C = 1													# Constraint on Lagrange multipliers alpha for L1 norm. Lower values imply stronger regularization

	## Init the training set
	num_obs = 10000
	x1 = np.random.uniform(-10, 10, num_obs)
	x2 = np.random.normal(0, 2, num_obs)
	x3 = np.random.normal(2, 10, num_obs) 
	x4 = np.random.chisquare(10, num_obs) 
	X = np.column_stack([x1, x2, x3, x4])
	print ("X.shape: ", X.shape)
	y = []
	for idx in range(num_obs) :
		y.append(np.sign(10 + x1[idx] + 4.2*x2[idx] - 3*x3[idx] + 1.5*math.log(x4[idx])))
	print ("y[0...10]: ", y[0:10])
	
	## Init the test set
	num_test_obs = 1000
	x1 = np.random.uniform(-8, 8, num_test_obs) 
	x2 = np.random.normal(0.5, 2.5, num_test_obs)
	x3 = np.random.normal(2, 11, num_test_obs)
	x4 = np.random.chisquare(9, num_test_obs) 
	X_test = np.column_stack([x1, x2, x3, x4])
	print ("X_test.shape: ", X_test.shape)
	y_test = []
	for idx in range(num_test_obs) :
		y_test.append(np.sign(10 + x1[idx] + 4.2*x2[idx] - 3*x3[idx] + 1.5*math.log(x4[idx])))
	print ("y_test.shape: ", len(y_test))
	print ("y_test[0...10]: ", y_test[0:10])

	## Standard scale the features prior to fitting
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	X_test = scaler.transform(X_test)

	clf = SVMClassifier(penalty, C = C, kernel = kernel, gamma = gamma)
	clf.fit(X, y)
	score = clf.score(X_test, y_test)
	print ("Prediction accuracy: %0.4f" %(score))

