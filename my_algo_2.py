#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import time
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import stats
from pylab import rcParams
from sklearn.utils import check_random_state
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split



# ================== Function to fetch the dataset =====================================
def download():
	"""Function to fetch the dataset
	
	Returns:
		TYPE: Datasets X and y
	"""
	filename='percent_inside_new.csv'
	df = pd.read_csv('Data/'+filename)
	
	mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9}
	df['labels'].replace(mapping,inplace=True)


	y = df['labels']
	print(df.head())

	df.drop('labels',axis=1,inplace=True)
	X = df
	print ('Dataset:', X.shape, y.shape)

	# Convert to numpy array
	X = X.values
	y = y.values

	return (X, y)

# =======================================================================

# =================== Function to split dataset into training and testing =======================
def split(train_size):
	"""Summary
	
	Args:
		train_size (TYPE): Size of the training set
	
	Returns:
		TYPE: Splitted train and test set
	"""
	X_train_full,X_test,y_train_full,y_test = train_test_split(X, y, train_size=0.9, shuffle=True)
	return (X_train_full, y_train_full, X_test, y_test)

# ================================================================================================

# ========== Super class for a model =========================================================
class BaseModel(object):

	"""Summary
	"""
	
	def __init__(self):
		"""Summary
		"""
		pass

	def fit_predict(self):
		"""Summary
		"""
		pass
# =============================================================================================

# ============== Classes for each model ==================================================

class SvmModel(BaseModel):

	"""Class for SVM classifier
	
	Attributes:
		classifier (TYPE): Description
		model_type (str): Description
		test_y_predicted (TYPE): Description
		val_y_predicted (TYPE): Description
	"""
	
	model_type = 'Support Vector Machine with linear Kernel'

	
	
	def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):
		"""Summary
		
		Args:
			X_train (TYPE): Training features
			y_train (TYPE): Training labels
			X_val (TYPE): Validation features
			X_test (TYPE): Testing features
			c_weight (TYPE): Class weight
		
		Returns:
			TYPE: Train test set and predicted values
		"""
		print ('training svm...')
		self.classifier = SVC(C=1, kernel='linear', probability=True,class_weight=c_weight)
		self.classifier.fit(X_train, y_train)
		self.test_y_predicted = self.classifier.predict(X_test)
		self.val_y_predicted = self.classifier.predict(X_val)
		return (X_train, X_val, X_test, self.val_y_predicted,self.test_y_predicted)


class LogModel(BaseModel):

	"""Class for logistic regression model
	
	Attributes:
		classifier (TYPE): Description
		model_type (str): Description
		test_y_predicted (TYPE): Description
		val_y_predicted (TYPE): Description
	"""
	
	model_type = 'Multinominal Logistic Regression' 
	
	def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):
		"""
		
		Args:
			X_train (TYPE): Training features
			y_train (TYPE): Training labels
			X_val (TYPE): Validation features
			X_test (TYPE): Testing features
			c_weight (TYPE): Class weight
		
		Returns:
			TYPE: Train test set and predicted values
		"""
		print ('training multinomial logistic regression')
		train_samples = X_train.shape[0]
		self.classifier = LogisticRegression(
			C=50. / train_samples,
			multi_class='multinomial',
			penalty='l1',
			solver='saga',
			tol=0.1,
			class_weight=c_weight,
			)

		self.classifier.fit(X_train, y_train)
		self.test_y_predicted = self.classifier.predict(X_test)
		self.val_y_predicted = self.classifier.predict(X_val)
		return (X_train, X_val, X_test, self.val_y_predicted,self.test_y_predicted)

class RfModel(BaseModel):

	"""Class for Random forest classifier
	
	Attributes:
		classifier (TYPE): Description
		model_type (str): Description
		test_y_predicted (TYPE): Description
		val_y_predicted (TYPE): Description
	"""
	
	model_type = 'Random Forest'
	
	def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):
		"""Summary
		
		Args:
			X_train (TYPE): Training features
			y_train (TYPE): Training labels
			X_val (TYPE): Validation features
			X_test (TYPE): Testing features
			c_weight (TYPE): Class weight
		
		Returns:
			TYPE: Train test set and predicted values
		"""
		print ('training random forest...')
		self.classifier = RandomForestClassifier(n_estimators=500, class_weight=c_weight)

		self.classifier.fit(X_train, y_train)
		self.test_y_predicted = self.classifier.predict(X_test)
		self.val_y_predicted = self.classifier.predict(X_val)
		return (X_train, X_val, X_test, self.val_y_predicted, self.test_y_predicted)

# =============================================================================================


# ============= Class to train the model ============================

class TrainModel:

	def __init__(self, model_object1, model_object2, model_object3):

		self.accuracies = []
		self.model_object1 = model_object1()
		self.model_object2 = model_object2()
		self.model_object3 = model_object3()       

	def print_model_type(self):
		print (self.model_object1.model_type)
		print (self.model_object1.model_type)
		print (self.model_object1.model_type)

	# we train normally and get probabilities for the validation set. i.e., we use the probabilities to select the most uncertain samples

	def train(self, X_train, y_train, X_val, X_test, c_weight):
		print ('Train set:', X_train.shape, 'y:', y_train.shape)
		print ('Val   set:', X_val.shape)
		print ('Test  set:', X_test.shape)
		t0 = time.time()
		(X_train, X_val, X_test, self.val_y_predicted1,self.test_y_predicted1) = self.model_object1.fit_predict(X_train, y_train, X_val, X_test, c_weight)
		(X_train, X_val, X_test, self.val_y_predicted2,self.test_y_predicted2) = self.model_object2.fit_predict(X_train, y_train, X_val, X_test, c_weight)
		(X_train, X_val, X_test, self.val_y_predicted3,self.test_y_predicted3) = self.model_object3.fit_predict(X_train, y_train, X_val, X_test, c_weight)


		self.test_y_predicted1 = self.model_object1.classifier.predict_proba(X_test)
		self.test_y_predicted2 = self.model_object2.classifier.predict_proba(X_test)
		self.test_y_predicted3 = self.model_object3.classifier.predict_proba(X_test)


		self.test_y_predicted = ensemble_probas(self.test_y_predicted1,self.test_y_predicted2,self.test_y_predicted3)
		self.test_y_predicted = np.argmax(self.test_y_predicted,axis=1)

		print('Y_predicted: ',self.test_y_predicted)
		
		self.run_time = time.time() - t0
		return (X_train, X_val, X_test)  # we return them in case we use PCA, with all the other algorithms, this is not needed.

	# we want accuracy only for the test set

	def get_test_accuracy(self, i, y_test):

		classif_rate = np.mean(self.test_y_predicted.ravel() == y_test.ravel()) * 100
		self.accuracies.append(classif_rate)               
		print('--------------------------------')
		print('Iteration:',i)
		print('Y_predicted test acc: ',self.test_y_predicted)
		print('--------------------------------')
		print('y-test set:',y_test.shape)
		print('Example run in %.3f s' % self.run_time,'\n')
		print("Accuracy rate for %f " % (classif_rate))    
		
		
		print('--------------------------------')

# =============================================================================================

# ================== Classes for selection ======================================
class BaseSelectionFunction(object):

	def __init__(self):
		pass

	def select(self):
		pass


class RandomSelection(BaseSelectionFunction):

	@staticmethod
	def select(probas_val, initial_labeled_samples):
		random_state = check_random_state(0)
		selection = np.random.choice(probas_val.shape[0], initial_labeled_samples, replace=False)
		return selection


class EntropySelection(BaseSelectionFunction):

	@staticmethod
	def select(probas_val, initial_labeled_samples):
		e = (-probas_val * np.log2(probas_val)).sum(axis=1)
		selection = (np.argsort(e)[::-1])[:initial_labeled_samples]
		return selection
	  
	  
class MarginSamplingSelection(BaseSelectionFunction):

	@staticmethod
	def select(probas_val, initial_labeled_samples):
		rev = np.sort(probas_val, axis=1)[:, ::-1]
		values = rev[:, 0] - rev[:, 1]
		selection = np.argsort(values)[:initial_labeled_samples]
		return selection

# ==========================================================================

# ================ Normalizer ==============================================
class Normalize(object):
	
	def normalize(self, X_train, X_val, X_test):
		self.scaler = MinMaxScaler()
		X_train = self.scaler.fit_transform(X_train)
		X_val   = self.scaler.transform(X_val)
		X_test  = self.scaler.transform(X_test)
		return (X_train, X_val, X_test) 
	
	def inverse(self, X_train, X_val, X_test):
		X_train = self.scaler.inverse_transform(X_train)
		X_val   = self.scaler.inverse_transform(X_val)
		X_test  = self.scaler.inverse_transform(X_test)
		return (X_train, X_val, X_test) 

# ============================================================================


# Function to initially select k random samples

def get_k_random_samples(initial_labeled_samples, X_train_full,y_train_full):
	random_state = check_random_state(0)
	permutation = np.random.choice(trainset_size,initial_labeled_samples,replace=False)
	print ()
	print ('initial random chosen samples', permutation.shape)
	X_train = X_train_full[permutation]
	y_train = y_train_full[permutation]
	X_train = X_train.reshape((X_train.shape[0], -1))
	bin_count = np.bincount(y_train.astype('int64'))
	unique = np.unique(y_train.astype('int64'))
	print (
		'initial train set:',
		X_train.shape,
		y_train.shape,
		'unique(labels):',
		bin_count,
		unique,
		)
	return (permutation, X_train, y_train)

# ============================================================================

# ================= The main algorithm =======================================

class TheAlgorithm(object):

	accuracies = []

	def __init__(self, initial_labeled_samples, model_object1,model_object2,model_object3, selection_function):
		self.initial_labeled_samples = initial_labeled_samples

		# === Models to be used ===============================
		self.model_object1 = model_object1
		self.model_object2 = model_object2
		self.model_object3 = model_object3
		# =====================================================

		self.sample_selection_function = selection_function

	def run(self, X_train_full, y_train_full, X_test, y_test):

		# initialize process by applying base learner to labeled training data set to obtain Classifier

		(permutation, X_train, y_train) = get_k_random_samples(self.initial_labeled_samples,X_train_full, y_train_full)
		self.queried = self.initial_labeled_samples
		self.samplecount = [self.initial_labeled_samples]


		X_val = np.array([])
		y_val = np.array([])
		X_val = np.copy(X_train_full)
		X_val = np.delete(X_val, permutation, axis=0)
		y_val = np.copy(y_train_full)
		y_val = np.delete(y_val, permutation, axis=0)
		print ('val set:', X_val.shape, y_val.shape, permutation.shape)
		print ()

		# normalize data

		normalizer = Normalize()
		X_train, X_val, X_test = normalizer.normalize(X_train, X_val, X_test)   
		
		# ============ train the model here =========================================
		self.clf_model = TrainModel(self.model_object1,self.model_object2,self.model_object3)
		(X_train, X_val, X_test) = self.clf_model.train(X_train, y_train, X_val, X_test, 'balanced')
		active_iteration = 1
		self.clf_model.get_test_accuracy(1, y_test)
		# ============================================================================

		while self.queried < max_queried:

			active_iteration += 1

			# get validation probabilities for all 3 classifiers
			# =====================================================================================
			probas_val1 = self.clf_model.model_object1.classifier.predict_proba(X_val)
			print ('val predicted:',self.clf_model.val_y_predicted1.shape,self.clf_model.val_y_predicted1)
			print ('probabilities:', probas_val1.shape, '\n',np.argmax(probas_val1, axis=1))


			probas_val2 = self.clf_model.model_object2.classifier.predict_proba(X_val)
			print ('val predicted:',self.clf_model.val_y_predicted2.shape,self.clf_model.val_y_predicted2)
			print ('probabilities:', probas_val2.shape, '\n',np.argmax(probas_val2, axis=1))

			probas_val3 = self.clf_model.model_object3.classifier.predict_proba(X_val)
			print ('val predicted:',self.clf_model.val_y_predicted3.shape,self.clf_model.val_y_predicted3)
			print ('probabilities:', probas_val3.shape, '\n',np.argmax(probas_val3, axis=1))
			# =====================================================================================

			# ==================== Calculate the ensemble probability ===============================
			probas_val = ensemble_probas(probas_val1=probas_val1,probas_val2=probas_val2,probas_val3=probas_val3)
			# =======================================================================================


			# select samples using a selection function

			uncertain_samples = self.sample_selection_function.select(probas_val, self.initial_labeled_samples)

			# normalization needs to be inversed and recalculated based on the new train and test set.
 
			X_train, X_val, X_test = normalizer.inverse(X_train, X_val, X_test)   

			# get the uncertain samples from the validation set

			print ('trainset before', X_train.shape, y_train.shape)
			X_train = np.concatenate((X_train, X_val[uncertain_samples]))
			y_train = np.concatenate((y_train, y_val[uncertain_samples]))
			print ('trainset after', X_train.shape, y_train.shape)
			self.samplecount.append(X_train.shape[0])

			bin_count = np.bincount(y_train.astype('int64'))
			unique = np.unique(y_train.astype('int64'))
			print (
				'updated train set:',
				X_train.shape,
				y_train.shape,
				'unique(labels):',
				bin_count,
				unique,
				)

			X_val = np.delete(X_val, uncertain_samples, axis=0)
			y_val = np.delete(y_val, uncertain_samples, axis=0)
			print ('val set:', X_val.shape, y_val.shape)
			print ()

			# normalize again after creating the 'new' train/test sets
			normalizer = Normalize()
			X_train, X_val, X_test = normalizer.normalize(X_train, X_val, X_test)               

			self.queried += self.initial_labeled_samples

			# ============ Retrain model ==================================================
			(X_train, X_val, X_test) = self.clf_model.train(X_train, y_train, X_val, X_test, 'balanced')
			self.clf_model.get_test_accuracy(1, y_test)
			# =============================================================================

		print ('final active learning accuracies',self.clf_model.accuracies)
		

# =========================================================================================================

# ================== Function for ensembling the probabilities =======================================
def ensemble_probas(probas_val1,probas_val2,probas_val3):
	res = (probas_val1+probas_val2+probas_val3)/3.0
	return res
# ====================================================================================================

# ================== Pickle the data ============================================
def pickle_save(fname, data):
  filehandler = open(fname,"wb")
  pickle.dump(data,filehandler)
  filehandler.close() 
  print('saved', fname, os.getcwd(), os.listdir())

def pickle_load(fname):
  print(os.getcwd(), os.listdir())
  file = open(fname,'rb')
  data = pickle.load(file)
  file.close()
  print(data)
  return data
# ===============================================================================


def experiment(d, models, selection_functions, Ks, repeats, contfrom):
	algos_temp = []
	print ('stopping at:', max_queried)
	count = 0

	for selection_function in selection_functions:
			d[selection_function.__name__] = {}
			
			for k in Ks:
				d[selection_function.__name__][str(k)] = []           
				
				for i in range(0, repeats):
					count+=1
					if count >= contfrom:
						print ('Count = %s, selection_function = %s, k = %s, iteration = %s.' % (count, selection_function.__name__, k, i))
						alg = TheAlgorithm(k, models[0],models[1],models[2], selection_function)
						alg.run(X_train_full, y_train_full, X_test, y_test)
						d[selection_function.__name__][str(k)].append(alg.clf_model.accuracies)
						fname = 'Active-learning-experiment-' + str(count) + '.pkl'
						pickle_save(fname, d)
						
						print(json.dumps(d, indent=2, sort_keys=True))
						print ()
						print ('---------------------------- FINISHED ---------------------------')
						print ()
	return d



(X, y) = download()

# ============= PARAMETERS WHICH CAN BE TUNED =========================================
max_queried = 500 # Determine the maximum number of queries that you want to carry out
total_size = X.shape[0]  # Total size of data
trainset_size = int(0.9*total_size) # Part of data to be taken for training
num_each_classes = total_size/10 # Number of classes in the data
Ks = [250] # Number of samples to select each time
repeats = 1

models = [SvmModel, RfModel, LogModel] 

selection_functions = [RandomSelection, MarginSamplingSelection, EntropySelection]
# ======================================================================================


(X_train_full, y_train_full, X_test, y_test) = split(trainset_size)
print ('train:', X_train_full.shape, y_train_full.shape)
print ('test :', X_test.shape, y_test.shape)
classes = len(np.unique(y))
print ('unique classes', classes)


d = {}
stopped_at = -1
d = experiment(d, models, selection_functions, Ks, repeats, stopped_at+1)

results = json.loads(json.dumps(d, indent=2, sort_keys=True))

with open('accuracies/acc.json', 'wb') as fp:
    json.dumps(d,fp,indent=2, sort_keys=True)

print(results)