"""
__Author__ : Vijay Sathish
__Date__	 : 08/17/2015

	- Decision Tree Classifier. Can perform multinomial classification
	- Two information gain criterion supported - gini impurity and entropy
	Dimensions:
		X - mxn
		y - mx1

	Implementation:
		- True branch is always the right branch
		- False branch is always the left branch
		- If a leaf node has multiple results that are possible, go with the result that has the maximum frequency
"""

import numpy as np
import math
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DecisionNode :
	def __init__(self, col_idx = None, col_val = None, true_branch = None, false_branch = None, result = None) :
		self.colIdx = col_idx
		self.colValue = col_val
		self.trueBranch = true_branch
		self.falseBranch = false_branch
		self.result = result

class DecisionTreeClassifier:
	def __init__(self, criterion = 'gini', max_depth = 3, min_samples_split = 2, min_samples_leaf = 1) :
		self.X = []
		self.y = []
		self.numObs = 0
		self.numFeatures = 0
		self.XTestNumObs = 0
		self.treeRoot = None											# Handle to the root of the decision tree after fit

		## Init model paramenters
		self.criterion = criterion
		self.maxDepth = max_depth
		self.minSamplesSplit = min_samples_split
		self.minSamplesLeaf = min_samples_leaf		# TODO: this stoppage criterion is not implemented

	#### BEGIN - INTERNAL HELPER METHODS ####

	## Helper function for gini and entropy calculations
	def compute_element_probabilities (self, elements) :
		num_elements = len(elements)
		
		## calculate probabilities for each unique element in input
		elems_dict = {}
		elems_prob = []
		for elem in np.unique(elements) :
			elem_count = len([1 for x in elements if x == elem])
			elems_dict[elem] = float(elem_count) / num_elements
			elems_prob.append(elems_dict[elem])
		# print ("Element probs: ", elems_dict)

		return elems_prob

	## Helper function to computeGiniInfoGain()
	def pairwiseGiniScore (self, elems_prob) :
		gini = 0.
		for i, prob_i in enumerate(elems_prob) :
			if i == len(elems_prob) - 1 :
				break
			for j in range(i+1, len(elems_prob)) :
				prob_j = elems_prob[j]
				gini += prob_i * prob_j
		return gini

	## Compute pairwise gini impurity for a list of elements
	def computeGiniInfoGain (self, y_true_branch, y_false_branch) :
		probs_tb = np.array(self.compute_element_probabilities(y_true_branch))
		probs_fb = np.array(self.compute_element_probabilities(y_false_branch))
		print ("probs_tb: ", probs_tb)
		print ("probs_fb: ", probs_fb)

		## Compute pair-wise gini from list of probabilities
		gini_tb = self.pairwiseGiniScore(probs_tb)
		gini_fb = self.pairwiseGiniScore(probs_fb)

		## Weight the entropy of each branch by the number of elements in the branch
		weight_tb = float(len(probs_tb)) / (len(probs_tb) + len(probs_fb))
		weight_fb = 1 - weight_tb
		info_gain = weight_tb * gini_tb + weight_fb * gini_fb
		return info_gain
	
	## Compute entropy for a list of elements
	def computeEntropyInfoGain (self, y_true_branch, y_false_branch) :
		probs_tb = np.array(self.compute_element_probabilities(y_true_branch))
		probs_fb = np.array(self.compute_element_probabilities(y_false_branch))

		## Compute entropy from list of probabilities
		entropy_tb = - np.dot(probs_tb, np.log2(probs_tb))
		entropy_fb = - np.dot(probs_fb, np.log2(probs_fb))

		## Weight the entropy of each branch by the number of elements in the branch
		weight_tb = float(len(probs_tb)) / (len(probs_tb) + len(probs_fb))
		weight_fb = 1 - weight_tb
		info_gain = weight_tb * entropy_tb + weight_fb * entropy_fb
		return info_gain

	## Helper to findBestSplit(): Divide a set of observations based on a column index and column value
	def divideSet (self, X_subset, y_subset, col_idx, col_val) :
		## set the behavior of function splitting based on whether variable is nominal or interval/ratio
		func_split_rows = None
		if (isinstance(col_val, int) or isinstance(col_val, float)) :
			func_split_rows = lambda x : x >= col_val
		else :
			func_split_rows = lambda x : x == col_val
		
		X_true_branch, X_false_branch, y_true_branch, y_false_branch = [], [], [], []
		for obs_idx in range(X_subset.shape[0]) :
			if func_split_rows(X_subset[obs_idx, col_idx]) is True :
				# print (X_subset[obs_idx, :])
				X_true_branch.append(X_subset[obs_idx, :])
				y_true_branch.append(y_subset[obs_idx])
			else :
				X_false_branch.append(X_subset[obs_idx, :])
				y_false_branch.append(y_subset[obs_idx])

		# print ("X_true_branch: ", X_true_branch)
		return X_true_branch, X_false_branch, y_true_branch, y_false_branch				
	
	## Helper function to buildTree. Returns count of each unique result in leaf node
	def uniqueCounts (self, y_subset) :
		results = {}
		for result in y_subset :
			if result not in results.keys() :
				results[result] = 0
			results[result] += 1
		return results

	## For a given set of observations, find and return the best split that maximizes information gain recursively
	def buildTree (self, X_subset, y_subset, depth) :
		print ("Depth_%d:" %(depth))
		print ("X_subset: ", X_subset)
		print ("y_subset: ", y_subset)
		## Init variables to store the best split
		best_gain = 0
		best_col_idx = None
		best_col_val = None
		best_tb = None				# True branch observations
		best_fb = None				# False branch observations

		## Check for early stoppage criterion
		if (len(y_subset) == 0) :
			return DecisionNode()

		## Iterate through each feature and each possible value for the feature to arrive at best split
		num_obs = len(y_subset)
		for col_idx in range(self.numFeatures) :
			print ("col_%d: " %(col_idx), np.unique(X_subset[:, col_idx]))
			for col_val in np.unique(X_subset[:, col_idx]) :

				X_true_branch, X_false_branch, y_true_branch, y_false_branch = self.divideSet (X_subset, y_subset, col_idx, col_val)
				if (len(y_true_branch) == 0 or len(y_false_branch) == 0) :
					continue				# If any branch has zero elements, that means we didn't really do any branching for this case
				
				if (criterion == 'gini') :
					info_gain = self.computeGiniInfoGain (y_true_branch, y_false_branch)
				else :
					assert(criterion == 'entropy')
					info_gain = self.computeEntropyInfoGain (y_true_branch, y_false_branch)
				
				## Update the best set after each iteration
				print ("info_gain: %0.3f; col_idx: %d; col_val: " %(info_gain, col_idx), col_val)
				if best_gain < info_gain :
					best_gain = info_gain
					best_col_idx = col_idx
					best_col_val = col_val
					best_tb = (X_true_branch, y_true_branch)
					best_fb = (X_false_branch, y_false_branch)
	
		if (best_gain > 0.) :
			print ("FINAL: best_gain: %0.3f; best_col_idx: %d; best_col_val: " %(best_gain, best_col_idx), best_col_val)
			print ("best_tb[0]: ", best_tb[0])
			print ("best_fb[0]: ", best_fb[0])
		else :
			print ("Zero info gain, no further splits possible!")
		# sys.exit()

		## Create the sub-branches recursively
		if best_gain > 0 and depth < self.maxDepth :
			if len(best_tb[1]) >= self.minSamplesSplit :
				true_branch = self.buildTree(np.array(best_tb[0]), best_tb[1], depth+1)
			else :
				true_branch = DecisionNode(result = self.uniqueCounts(best_tb[1]))
			
			if len(best_fb[1]) >= self.minSamplesSplit :
				false_branch = self.buildTree(np.array(best_fb[0]), best_fb[1], depth+1)
			else :
				false_branch = DecisionNode(result = self.uniqueCounts(best_fb[1]))
			return DecisionNode(best_col_idx, best_col_val, true_branch, false_branch)

		else :
			return DecisionNode(result = self.uniqueCounts(y_subset))

	## Helper to predict to figure out which branch to follow
	def followTrueBranch (self, X, col_idx, col_val) :
		## set the behavior of function splitting based on whether variable is nominal or interval/ratio
		func_split_rows = None
		if (isinstance(col_val, int) or isinstance(col_val, float)) :
			func_split_rows = lambda x : x >= col_val
		else :
			func_split_rows = lambda x : x == col_val
		return func_split_rows(X[col_idx])

	## Helper to predict to select most likely result
	def computeMostLikelyResult (self, results_dict) :
		total = np.array([value for value in results_dict.values()])
		total = np.sum(total)
		best_prob = 0.
		best_result = None
		for result in results_dict.keys() :
			result_prob = float(results_dict[result]) / total
			if (best_prob < result_prob) :
				best_prob = result_prob
				best_result = result
		print ("Best result: %s; prob: %0.3f" %(best_result, best_prob))
		return best_result
			
	#### END - INTERNAL HELPER METHODS ####

	def fit (self, X, y) :
		self.X = X																							# m x n
		self.numObs, self.numFeatures = X.shape
		self.y = y
		print ("self.y.shape: ", self.y.shape)
		
		## Recursively split tree until stopping criteria reached
		self.treeRoot = self.buildTree (X, y, depth = 0)

	def predict (self, X_test) :
		print ("\n\nPREDICTION PHASE...")
		self.XTestNumObs = X_test.shape[0]
		assert (X_test.shape[1] == self.numFeatures)

		## For each observation, follow the decision tree till you reach a leaf node and return the result
		y_pred = []
		for obs_idx in range(X_test.shape[0]) :
			node = self.treeRoot
			depth = 0
			while (1) :
				if node.result :
					print ("Results: ", node.result)
					break
				else :
					decision = self.followTrueBranch (X_test[obs_idx, :], node.colIdx, node.colValue)
					depth += 1
					print ("Following %d branch for depth = %d" %(decision, depth))
					if (decision is True) :
						node = node.trueBranch
					else :
						node = node.falseBranch
			result = self.computeMostLikelyResult(node.result)
			y_pred.append(result)
		print ("y_pred: ", y_pred)
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
			print ("Obs_%d, y_test: %s, y_pred: %s" %(test_idx, y_test[test_idx], y_pred[test_idx]))
		return (acc_count / X_test.shape[0])


if __name__ == '__main__':
	## Set random seed
	np.random.seed(1234)

	## Set model parameters
	criterion = 'gini'
	min_samples_split = 2
	min_samples_leaf = 2
	max_depth = 5

	## Init the training data
	train_data = pd.read_csv("D:/MLModels/tree_data.csv")
	print ("Header: ", list(train_data))

	y = train_data['subscription'].values
	X = np.array(train_data.drop(['subscription'], axis = 1))

	## Init the classifier
	clf = DecisionTreeClassifier(criterion, max_depth, min_samples_split, min_samples_leaf)
	clf.fit(X, y)
	score = clf.score(X, y)
	print ("Prediction accuracy: %0.4f" %(score))

