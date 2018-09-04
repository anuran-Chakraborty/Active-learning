
# coding: utf-8

# [View in Colaboratory](https://colab.research.google.com/github/orico/ActiveLearningFrameworkTutorial/blob/master/Active_Learning_Tutorial.ipynb)

# ## **Active Learning Tutorial**
# 
# These days we are exposed to an abundance of unlabeled data either from the Internet or from some other source such as academia or business worlds. Due to the fact that unlabeled data is relatively easy to acquire and is expensive to label, companies usually employ an expert or several employees whose purpose is to label data [1]. Consider the following situation, a data-driven medical company has a lot of MRI scans and they need to employ an expert that will help them interpret these scans. The company has limited resources and they cant interpret or label all of their data; this is the point where they decide to use active-learning (AL). The promise of AL is that by iteratively increasing the size of our carefully selected labeled data, it is possible to achieve similar (or greater [2]) performance to using a fully supervised data-set with a fraction of the cost or time that it takes to label all the data. AL is considered to be a semi-supervised method, between unsupervised and fully supervised in terms of the amount of labeled data, i.e., for unsupervised data we use 0% labeled samples and for fully supervised we use 100% labeled samples. Therefore, the decision of how much data to use or alternatively how much performance is required from the model relies on a resource management decision, in other words it can be a business decision. 
# 
# There are three scenarios for AL: 
# 1. Membership query synthesis, i.e., a generated sample is sent to an oracle for labeling.
# 2. Stream-Based selective sampling, i.e, each sample is considered separately - in our case for lable-querying or rejection. Similarly to online-learning, the data is not saved, there are no assumptions on data distribution, and therefore it is adaptive to change. 
# 3. Pool-Based sampling, i.e., sampled are chosen from a pool of unlabeled data for the purpose of labeling [3]. 
# In this tutorial we use the third scenario.
# 
# The following pseudo algorithm represents the learning process, as written in the code, for pool-based sampling:
# 1. Divide the data to a 'pool' and a test-set
# 2. Select 'k' samples from the pool for the initial train-set and label them, the remaining data will be the validation-set
# 3. Normalize all the sets
# 4. Train the Model using the train-set, with balanced weights.
# 5. Use the trained model with the validation-set, get probabilities per sample.
# 6. Use the trained model with the test-set, get performance measures.
# 7. Select 'k' most-informative samples based on per-sample-probabilities, i.e., those that the model was most uncertain about regarding their labelling.
# 8. Move these 'k' samples from the validation set to the train-set and query their labels.
# 9. Inverse normalization for all the data-sets
# 10. Stop according to the stop criterion, otherwise go to 3. 
# 
# There are a few things to note before going forward: 
# 1. The fully-supervised performance of a chosen algorithm is usually the upper bound, therefore it is advisable to try several algorithms.
# 2. Normalization for all sets must be inversed and normalized again after we remove samples from the validation set, because our sample distribution changed in both the new validation and new train-sets.
# 2. The sample selection function relies on test-sample probabilities derived from the trained model, therefore we can only use algorithms that provide access to sample probabilities.
# 3. 'k' is a hyper parameter
# 
# Our most important tool in AL method is the sample selection function, this is the only point where we influence the learning process and it crucial to use the right method. This area is a hot research topic and there are many studies that propose competing selection functions. 
# In this tutorial I propose four known selection functions:
# 1. Random selection - we select 'k' random samples from the validation set.
# 2. Entropy selection - we select 'k' samples with the highest entropy, i.e., with high uncertainty.
# 3. Margin selection - we select 'k' samples with the lowest difference between the two highest class probabilities, i.e., a higher figure will be given for samples whose model was very certain about a single class and lower to samples whose class probabilities are very similar. 
# 
# The code provided [here](https://github.com/orico/ActiveLearningFrameworkTutorial) utilizes a modular architecture in terms of selecting various learning algorithms and selection functions and can be used as a base for other model-function comparisons.
# 
# We compare several learning algorithms, such as support vector machine (SVM) with a linear kernel, random forest (RF) and logistic regression (LOG). Each algorithm was executed with all of the selection functions using all 'k' = [10,25,50,125,250], accumulating a total of 80 experiments. Due to the random nature of some of the algorithms and the selection functions, it is advisable to run repeated experiments in the code in order to calculate a statistical significant result. However, running times are long and I have chosen to run the experiment only once for each combination of (model,function,k).
# 
# The following is an explanation of the code and its class architecture.
# 

# We start with all the needed dependencies.

# In[ ]:


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

max_queried = 500


# We start by downloading our data and splitting it to train and test, according to known MNIST definitions 60K/10K split. later the train-set will be split to train and validation.
# 

# In[ ]:


total_size = 5000  # ie., testset_size = 10000
trainset_size = int(0.9*total_size) 
num_each_classes = total_size/10

def download():
    mnist = fetch_mldata('MNIST original')
    X = mnist.data.astype('float64')
    y = mnist.target.astype('int64')

    print('Dataset X: ',X.shape)
    print('Dataset Y: ',y.shape)

    
    count=0
    counteach=0
    classes=0

    x_new=np.zeros((total_size,784))
    y_new=np.zeros((total_size,))
    #=============== Changing dataset size ==========================
    while(classes<10):
        for i in range(70000):

            if(counteach==num_each_classes):
                classes=classes+1
                counteach=0

            if(y[i]==classes):
                x_new[count]=X[i]
                y_new[count]=y[i]
                count=count+1
                counteach=counteach+1


    #================================================================

    X=x_new
    y=y_new

    unique, counts = np.unique(y, return_counts=True)

    print (np.asarray((unique, counts)).T)


    print ('MNIST:', X.shape, y.shape)
    return (X, y)


def split(train_size):
    # X_train_full = X[:train_size]
    # y_train_full = y[:train_size]
    # X_test = X[train_size:]
    # y_test = y[train_size:]
    X_train_full,X_test,y_train_full,y_test = train_test_split(X, y, train_size=0.9, shuffle=True)
    return (X_train_full, y_train_full, X_test, y_test)


# We create a modular class representation, 'BaseModel' is a base model for the class architecture, you can implement new models and use them interchangeably or in addition to all other models.
# our current implementations include SVM, logistic regression, random forest and gradient boosting.

# In[ ]:


class BaseModel(object):

    def __init__(self):
        pass

    def fit_predict(self):
        pass


class SvmModel(BaseModel):

    model_type = 'Support Vector Machine with linear Kernel'
    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):
        print ('training svm...')
        self.classifier = SVC(C=1, kernel='linear', probability=True,
                              class_weight=c_weight)
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_test, self.val_y_predicted,
                self.test_y_predicted)


class LogModel(BaseModel):

    model_type = 'Multinominal Logistic Regression' 
    
    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):
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
        return (X_train, X_val, X_test, self.val_y_predicted,
                self.test_y_predicted)

class RfModel(BaseModel):

    model_type = 'Random Forest'
    
    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):
        print ('training random forest...')
        self.classifier = RandomForestClassifier(n_estimators=500, class_weight=c_weight)
        self.classifier.fit(X_train, y_train)
        self.test_y_predicted = self.classifier.predict(X_test)
        self.val_y_predicted = self.classifier.predict(X_val)
        return (X_train, X_val, X_test, self.val_y_predicted, self.test_y_predicted)
   


# Our 'TrainModel' class accepts one of the previously in defined learning algorithms, trains using the training set and gets performance measurements from the test set.

# In[ ]:


class TrainModel:

    def __init__(self, model_object):        
        self.accuracies = []
        self.model_object = model_object()        

    def print_model_type(self):
        print (self.model_object.model_type)

    # we train normally and get probabilities for the validation set. i.e., we use the probabilities to select the most uncertain samples

    def train(self, X_train, y_train, X_val, X_test, c_weight):
        print ('Train set:', X_train.shape, 'y:', y_train.shape)
        print ('Val   set:', X_val.shape)
        print ('Test  set:', X_test.shape)
        t0 = time.time()
        (X_train, X_val, X_test, self.val_y_predicted,
         self.test_y_predicted) = \
            self.model_object.fit_predict(X_train, y_train, X_val, X_test, c_weight)
        self.run_time = time.time() - t0
        return (X_train, X_val, X_test)  # we return them in case we use PCA, with all the other algorithms, this is not needed.

    # we want accuracy only for the test set

    def get_test_accuracy(self, i, y_test):
        classif_rate = np.mean(self.test_y_predicted.ravel() == y_test.ravel()) * 100
        self.accuracies.append(classif_rate)               
        print('--------------------------------')
        print('Iteration:',i)
        print('--------------------------------')
        print('y-test set:',y_test.shape)
        print('Example run in %.3f s' % self.run_time,'\n')
        print("Accuracy rate for %f " % (classif_rate))    
        print("Classification report for classifier %s:\n%s\n" % (self.model_object.classifier, metrics.classification_report(y_test, self.test_y_predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, self.test_y_predicted))
        print('--------------------------------')


# We create a modular selection function class representation, 'BaseSelectionFunction' is a base class for various sample selection methods. Using this architecture, you can implement new selection methods and use them in addition or instead of previous methods, for experimental purposes. Our current implementations include random-selection, entropy-selection, margin sampling-selection and minimum standard deviation-selection.

# In[ ]:


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

#     print('uniques chosen:',np.unique(selection).shape[0],'<= should be equal to:',initial_labeled_samples)

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


# We have a class that is used to normalize using a MinMax Scaler in the range of [0,1].

# In[ ]:


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


# Initially we would like to get a random sampling from the unlabeled data-pool, this is done using random.choice without replacement.

# In[ ]:


def get_k_random_samples(initial_labeled_samples, X_train_full,
                         y_train_full):
    random_state = check_random_state(0)
    permutation = np.random.choice(trainset_size,
                                   initial_labeled_samples,
                                   replace=False)
    print ()
    print ('initial random chosen samples', permutation.shape),
#            permutation)
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


# This is the main class that initiates the active-learning process according to the algorithm described in the introduction. In short, we select 'k' random samples, train a model, select the most informative samples, remove from the validation set, query their labels and retrain using those samples until reaching the stop criteria.

# In[ ]:


class TheAlgorithm(object):

    accuracies = []

    def __init__(self, initial_labeled_samples, model_object, selection_function):
        self.initial_labeled_samples = initial_labeled_samples
        self.model_object = model_object
        self.sample_selection_function = selection_function

    def run(self, X_train_full, y_train_full, X_test, y_test):

        # initialize process by applying base learner to labeled training data set to obtain Classifier

        (permutation, X_train, y_train) =             get_k_random_samples(self.initial_labeled_samples,
                                 X_train_full, y_train_full)
        self.queried = self.initial_labeled_samples
        self.samplecount = [self.initial_labeled_samples]

        # permutation, X_train, y_train = get_equally_k_random_samples(self.initial_labeled_samples,classes)

        # assign the val set the rest of the 'unlabelled' training data

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
        
        self.clf_model = TrainModel(self.model_object)
        (X_train, X_val, X_test) = self.clf_model.train(X_train, y_train, X_val, X_test, 'balanced')
        active_iteration = 1
        self.clf_model.get_test_accuracy(1, y_test)

        # fpfn = self.clf_model.test_y_predicted.ravel() != y_val.ravel()
        # print(fpfn)
        # self.fpfncount = []
        # self.fpfncount.append(fpfn.sum() / y_test.shape[0] * 100)

        while self.queried < max_queried:

            active_iteration += 1

            # get validation probabilities

            probas_val =                 self.clf_model.model_object.classifier.predict_proba(X_val)
            print ('val predicted:',
                   self.clf_model.val_y_predicted.shape,
                   self.clf_model.val_y_predicted)
            print ('probabilities:', probas_val.shape, '\n',
                   np.argmax(probas_val, axis=1))

            # select samples using a selection function

            uncertain_samples =                 self.sample_selection_function.select(probas_val, self.initial_labeled_samples)

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
            (X_train, X_val, X_test) = self.clf_model.train(X_train, y_train, X_val, X_test, 'balanced')
            self.clf_model.get_test_accuracy(active_iteration, y_test)

        print ('final active learning accuracies',
               self.clf_model.accuracies)




# We download the data, split to train validation and test, we run the experiment by iterating over all of our training algorithms X all of our selection functions X all possible k's in the range of [10,25,50,125,250]. The accuracy results are kept in a dictionary and pickle-saved to a unique file as soon as the model finishes training - this is crucial when using google colaboratory as it tends to disconnect from time to time. We also limit our training to a maximum of 500 queried samples.

# In[ ]:


(X, y) = download()
(X_train_full, y_train_full, X_test, y_test) = split(trainset_size)
print ('train:', X_train_full.shape, y_train_full.shape)
print ('test :', X_test.shape, y_test.shape)
classes = len(np.unique(y))
print ('unique classes', classes)

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
  
def experiment(d, models, selection_functions, Ks, repeats, contfrom):
    algos_temp = []
    print ('stopping at:', max_queried)
    count = 0
    for model_object in models:
      if model_object.__name__ not in d:
          d[model_object.__name__] = {}
      
      for selection_function in selection_functions:
        if selection_function.__name__ not in d[model_object.__name__]:
            d[model_object.__name__][selection_function.__name__] = {}
        
        for k in Ks:
            d[model_object.__name__][selection_function.__name__][str(k)] = []           
            
            for i in range(0, repeats):
                count+=1
                if count >= contfrom:
                    print ('Count = %s, using model = %s, selection_function = %s, k = %s, iteration = %s.' % (count, model_object.__name__, selection_function.__name__, k, i))
                    alg = TheAlgorithm(k, 
                                       model_object, 
                                       selection_function
                                       )
                    alg.run(X_train_full, y_train_full, X_test, y_test)
                    d[model_object.__name__][selection_function.__name__][str(k)].append(alg.clf_model.accuracies)
                    fname = 'Active-learning-experiment-' + str(count) + '.pkl'
                    pickle_save(fname, d)
                    if count % 5 == 0:
                        print(json.dumps(d, indent=2, sort_keys=True))
                    print ()
                    print ('---------------------------- FINISHED ---------------------------')
                    print ()
    return d


max_queried = 500 

repeats = 1

models = [SvmModel, RfModel, LogModel] 

selection_functions = [RandomSelection, MarginSamplingSelection, EntropySelection] 

Ks = [250,125,50,25,10] 

d = {}
stopped_at = -1 

# print('directory dump including pickle files:', os.getcwd(), np.sort(os.listdir()))  
# d = pickle_load('Active-learning-experiment-' + str(stopped_at) + '.pkl')  
# print(json.dumps(d, indent=2, sort_keys=True))

d = experiment(d, models, selection_functions, Ks, repeats, stopped_at+1)
print (d)
results = json.loads(json.dumps(d, indent=2, sort_keys=True))
print(results)


# Independently, we trained several models using a train-test split of 60K-10K, the results indicate that the upper-bound for RF, SVM and LOG are 97., 94. and 92.47, respectively.
# 
# The following graphs show that the random forest classifier paired with the margin-selection method and k=10 is the best configuration.

# In[64]:


def performance_plot(fully_supervised_accuracy, dic, models, selection_functions, Ks, repeats):  
    fig, ax = plt.subplots()
    ax.plot([0,500],[fully_supervised_accuracy, fully_supervised_accuracy],label = 'algorithm-upper-bound')
    for model_object in models:
      for selection_function in selection_functions:
        for idx, k in enumerate(Ks):
            x = np.arange(float(Ks[idx]), 500 + float(Ks[idx]), float(Ks[idx]))            
            Sum = np.array(dic[model_object][selection_function][k][0])
            for i in range(1, repeats):
                Sum = Sum + np.array(dic[model_object][selection_function][k][i])
            mean = Sum / repeats
            ax.plot(x, mean ,label = model_object + '-' + selection_function + '-' + str(k))
    ax.legend()
    ax.set_xlim([50,500])
    ax.set_ylim([40,100])
    ax.grid(True)
    plt.show()

models_str = ['SvmModel', 'RfModel', 'LogModel']
selection_functions_str = ['RandomSelection', 'MarginSamplingSelection', 'EntropySelection']
Ks_str = ['250','125','50','25','10'] 
repeats = 1
random_forest_upper_bound = 97.
svm_upper_bound = 94.
log_upper_bound = 92.47
total_experiments = len(models_str) * len(selection_functions_str) * len(Ks_str) * repeats

print('So which is the better model? under the stopping condition and hyper parameters - random forest is the winner!')
performance_plot(random_forest_upper_bound, d, ['RfModel'] , selection_functions_str    , Ks_str, 1)
performance_plot(svm_upper_bound, d, ['SvmModel'] , selection_functions_str    , Ks_str, 1)
performance_plot(log_upper_bound, d, ['LogModel'] , selection_functions_str    , Ks_str, 1)


# In[62]:



print('So which is the best sample selection function? margin sampling is the winner!')
performance_plot(random_forest_upper_bound, d, ['RfModel'], selection_functions_str    , Ks_str, 1)
print()
print('So which is the best k? k=10 is the winner')
performance_plot(random_forest_upper_bound, d, ['RfModel'] , ['MarginSamplingSelection'], Ks_str, 1)
