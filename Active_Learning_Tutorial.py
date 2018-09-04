
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
from sklearn.ensemble import RandomForestClassifier,     GradientBoostingClassifier

max_queried = 500


# We start by downloading our data and splitting it to train and test, according to known MNIST definitions 60K/10K split. later the train-set will be split to train and validation.
# 

# In[ ]:


trainset_size = 60000  # ie., testset_size = 10000

def download():
    mnist = fetch_mldata('MNIST original')
    X = mnist.data.astype('float64')
    y = mnist.target
    print ('MNIST:', X.shape, y.shape)
    return (X, y)


def split(train_size):
    X_train_full = X[:train_size]
    y_train_full = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
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


# I would like to thank Moshe Hadad for his valueable critique regarding PEP8 and Shay Zweig for his proof-reading and comments.
# 
# Ori Cohen has done his PhD in computer science in the fields of machine learning, brain-computer-interface and neurobiology.
# 
# [1] Shay Yehezkel, *High Dimensional Statistical Process Control and Application*, M.Sc Thesis.
# 
# [2] Ilhan, Hamza Osman, and Mehmet Fatih Amasyali. "[*Active Learning as a Way of Increasing Accuracy*](http://www.ijcte.org/papers/910-AC0013.pdf)." International Journal of Computer Theory and Engineering 6, no. 6 (2014): 460.
# 
# [3] Stefan Hosein [*Active Learning: Curious AI Algorithms*](https://www.datacamp.com/community/tutorials/active-learning)
# 

# In[5]:


#@title Executed code for the experiment with output.
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
from sklearn.ensemble import RandomForestClassifier,     GradientBoostingClassifier

trainset_size = 60000  # ie., testset_size = 10000
max_queried = 500


# ==============================================================================

def download():
    mnist = fetch_mldata('MNIST original')
    X = mnist.data.astype('float64')
    y = mnist.target
    print ('MNIST:', X.shape, y.shape)
    return (X, y)


def split(train_size):
    X_train_full = X[:train_size]
    y_train_full = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    return (X_train_full, y_train_full, X_test, y_test)


# ==============================================================================

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


class GmmModel(BaseModel):

    model_type = 'Gaussian Mixture Model'
    
    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):
        print ('training gaussian mixture model...')
        pca = PCA(n_components=75).fit(X_train)  # ,whiten=True).fit(X_train)
        reduced_train_data = pca.transform(X_train)
        reduced_test_data = pca.transform(X_test)
        reduced_val_data = pca.transform(X_val)
        print ('PCA: explained_variance_ratio_',
               np.sum(pca.explained_variance_ratio_))
        self.classifier = GaussianMixture(n_components=10, covariance_type='full')
        self.classifier.fit(reduced_train_data)
        self.test_y_predicted =             self.classifier.predict(reduced_test_data)
        self.val_y_predicted = self.classifier.predict(reduced_val_data)
        return (reduced_train_data, reduced_val_data,
                reduced_test_data, self.val_y_predicted,
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


class GbcModel(BaseModel):

    model_type = 'Gradient Boosting Classifier'
    
    def fit_predict(self, X_train, y_train, X_val, X_test, c_weight):
        print ('training gradient boosting...')
        parm = {
            'n_estimators': 1200,
            'max_depth': 3,
            'subsample': 0.5,
            'learning_rate': 0.01,
            'min_samples_leaf': 1,
            'random_state': 3,
            }
        self.classifier = GradientBoostingClassifier(**parm)
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


# ====================================================================================================

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


# ====================================================================================================

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


# ====================================================================================================

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


class MinStdSelection(BaseSelectionFunction):

    # select the samples where the std is smallest - i.e., there is uncertainty regarding the relevant class
    # and then train on these "hard" to classify samples.

    @staticmethod
    def select(probas_val, initial_labeled_samples):
        std = np.std(probas_val * 100, axis=1)
        selection = std.argsort()[:initial_labeled_samples]
        selection = selection.astype('int64')

#     print('std',std.shape,std)
#     print()
#     print('selection',selection, selection.shape, std[selection])

        return selection


class MarginSamplingSelection(BaseSelectionFunction):

    @staticmethod
    def select(probas_val, initial_labeled_samples):
        rev = np.sort(probas_val, axis=1)[:, ::-1]
        values = rev[:, 0] - rev[:, 1]
        selection = np.argsort(values)[:initial_labeled_samples]
        return selection


class EntropySelection(BaseSelectionFunction):

    @staticmethod
    def select(probas_val, initial_labeled_samples):
        e = (-probas_val * np.log2(probas_val)).sum(axis=1)
        selection = (np.argsort(e)[::-1])[:initial_labeled_samples]
        return selection


# ====================================================================================================

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

      
# ====================================================================================================

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


# get MNIST

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
            d[model_object.__name__][selection_function.__name__][k] = []           
            
            for i in range(0, repeats):
                count+=1
                if count >= contfrom:
                    print ('Count = %s, using model = %s, selection_function = %s, k = %s, iteration = %s.' % (count, model_object.__name__, selection_function.__name__, k, i))
                    alg = TheAlgorithm(k, 
                                       model_object, 
                                       selection_function
                                       )
                    alg.run(X_train_full, y_train_full, X_test, y_test)
                    d[model_object.__name__][selection_function.__name__][k].append(alg.clf_model.accuracies)
                    fname = 'Active-learning-experiment-' + str(count) + '.pkl'
                    pickle_save(fname, d)
                    if count % 5 == 0:
                        print(json.dumps(d, indent=2, sort_keys=True))
                    print ()
                    print ('---------------------------- FINISHED ---------------------------')
                    print ()
    return d

max_queried = 500
# max_queried = 20

repeats = 1

models = [SvmModel, RfModel, LogModel]#, GbcModel]
# models = [RfModel, SvmModel]

selection_functions = [RandomSelection, MarginSamplingSelection, EntropySelection]#, MinStdSelection]
# selection_functions = [MarginSamplingSelection]

Ks = [250,125,50,25,10]
# Ks = [10]

d = {}
stopped_at = -1

# stopped_at = 73
# d = pickle_load('Active-learning-experiment-'+ str(stopped_at) +'.pkl')  
# print(json.dumps(d, indent=2, sort_keys=True))

d = experiment(d, models, selection_functions, Ks, repeats, stopped_at+1)
print(json.dumps(d, indent=2, sort_keys=True))


# In[ ]:


#@title Manual assignment of the dictionary json dump to a variable, these are the results from the previous cell.

d = {
  "LogModel": {
    "EntropySelection": {
      "10": [
        [
          36.620000000000005,
          46.79,
          47.49,
          49.96,
          51.59,
          54.25,
          49.43,
          55.510000000000005,
          52.88,
          50.89,
          53.04,
          52.42,
          52.410000000000004,
          50.949999999999996,
          48.699999999999996,
          48.89,
          51.160000000000004,
          47.68,
          46.1,
          48.91,
          50.739999999999995,
          53.080000000000005,
          54.790000000000006,
          50.88,
          53.790000000000006,
          56.05,
          54.06999999999999,
          57.85,
          54.279999999999994,
          55.86,
          57.43000000000001,
          47.57,
          53.480000000000004,
          55.21,
          53.32,
          54.98,
          52.51,
          54.39000000000001,
          55.88999999999999,
          54.61,
          55.69,
          54.510000000000005,
          52.559999999999995,
          53.66,
          56.089999999999996,
          54.230000000000004,
          51.77,
          52.629999999999995,
          55.00000000000001,
          48.97
        ]
      ],
      "25": [
        [
          59.050000000000004,
          52.849999999999994,
          54.71,
          55.16,
          52.349999999999994,
          53.94,
          53.43,
          51.480000000000004,
          45.94,
          48.92,
          48.55,
          44.51,
          50.72,
          48.29,
          49.69,
          53.61,
          49.91,
          50.12,
          50.59,
          52.66
        ]
      ],
      "50": [
        [
          61.82,
          60.31999999999999,
          54.03,
          57.220000000000006,
          58.67,
          57.11000000000001,
          53.18000000000001,
          57.76,
          55.08,
          53.42
        ]
      ],
      "125": [
        [
          69.07,
          62.660000000000004,
          58.93000000000001,
          60.5
        ]
      ],
      "250": [
        [
          73.77,
          63.23
        ]
      ]
    },
    "MarginSamplingSelection": {
      "10": [
        [
          33.269999999999996,
          56.63,
          57.709999999999994,
          61.82,
          64.57000000000001,
          67.43,
          69.24,
          69.76,
          69.87,
          71.05,
          73.27,
          71.35000000000001,
          70.75,
          72.61,
          71.23,
          72.57000000000001,
          72.38,
          73.17,
          73.94,
          73.44000000000001,
          72.26,
          72.78,
          73.68,
          72.75,
          73.79,
          73.91,
          72.39,
          71.72,
          74.39,
          73.09,
          73.39,
          73.25,
          72.8,
          74.51,
          72.15,
          72.08,
          71.78,
          73.69,
          73.19,
          73.00999999999999,
          72.68,
          71.67,
          72.71,
          74.11,
          73.06,
          74.33999999999999,
          72.56,
          72.64,
          72.35000000000001,
          72.86
        ]
      ],
      "25": [
        [
          41.980000000000004,
          52.12,
          65.4,
          64.84,
          65.10000000000001,
          64.14,
          65.86999999999999,
          66.99000000000001,
          70.11,
          70.44,
          67.21000000000001,
          71.17,
          70.33,
          70.12,
          71.6,
          70.3,
          68.16,
          69.51,
          69.59,
          70.85000000000001
        ]
      ],
      "50": [
        [
          62.8,
          70.19,
          68.39,
          71.65,
          69.44,
          70.94,
          70.7,
          70.14,
          71.17,
          70.21
        ]
      ],
      "125": [
        [
          71.11,
          72.42,
          72.0,
          72.81
        ]
      ],
      "250": [
        [
          73.76,
          73.76
        ]
      ]
    },
    "RandomSelection": {
      "10": [
        [
          31.61,
          50.36000000000001,
          53.1,
          59.74,
          62.7,
          64.03999999999999,
          64.25,
          61.629999999999995,
          66.47,
          68.0,
          65.75999999999999,
          65.0,
          65.27,
          66.46,
          66.99000000000001,
          66.86,
          67.28,
          66.72,
          67.10000000000001,
          68.33,
          69.25,
          69.69999999999999,
          69.32000000000001,
          67.72,
          69.87,
          68.69,
          68.83,
          70.00999999999999,
          69.65,
          70.19,
          71.41,
          70.53,
          70.15,
          68.74,
          70.0,
          68.24,
          68.92,
          67.91,
          69.17999999999999,
          68.87,
          69.02000000000001,
          70.00999999999999,
          70.47,
          71.83,
          70.78,
          70.61,
          71.33,
          70.91,
          71.0,
          71.78999999999999
        ]
      ],
      "25": [
        [
          46.63,
          57.86,
          62.019999999999996,
          70.61,
          72.74000000000001,
          69.82000000000001,
          69.6,
          70.81,
          70.82000000000001,
          69.49,
          71.46000000000001,
          71.48,
          72.04,
          72.11,
          72.65,
          73.09,
          73.72,
          73.86,
          74.69,
          73.72999999999999
        ]
      ],
      "50": [
        [
          68.38,
          66.14999999999999,
          70.45,
          72.68,
          72.31,
          72.33000000000001,
          73.44000000000001,
          71.78999999999999,
          73.11999999999999,
          70.84
        ]
      ],
      "125": [
        [
          68.95,
          68.53,
          71.67,
          73.09
        ]
      ],
      "250": [
        [
          73.11,
          74.56
        ]
      ]
    }
  },
  "RfModel": {
    "EntropySelection": {
      "10": [
        [
          37.519999999999996,
          42.53,
          51.42,
          49.03,
          49.25,
          52.01,
          51.41,
          51.89,
          53.04,
          55.48,
          56.989999999999995,
          56.720000000000006,
          56.93,
          55.169999999999995,
          56.26,
          55.64,
          54.769999999999996,
          54.98,
          54.06999999999999,
          53.71,
          53.21,
          52.800000000000004,
          53.54,
          52.72,
          52.65,
          51.849999999999994,
          51.67,
          51.89,
          52.54,
          53.03,
          56.06,
          56.26,
          55.669999999999995,
          60.17,
          60.480000000000004,
          59.9,
          58.97,
          60.019999999999996,
          59.81999999999999,
          59.86,
          59.95,
          62.970000000000006,
          63.7,
          62.94,
          62.19,
          62.019999999999996,
          63.89,
          62.150000000000006,
          60.17,
          62.39
        ]
      ],
      "25": [
        [
          44.21,
          39.2,
          37.57,
          35.82,
          36.120000000000005,
          33.64,
          32.09,
          32.1,
          32.28,
          30.759999999999998,
          31.230000000000004,
          29.34,
          26.419999999999998,
          27.05,
          26.58,
          23.23,
          24.0,
          23.7,
          21.58,
          23.84
        ]
      ],
      "50": [
        [
          63.85999999999999,
          60.309999999999995,
          59.760000000000005,
          59.89,
          59.160000000000004,
          59.519999999999996,
          57.9,
          59.74,
          57.06,
          59.91
        ]
      ],
      "125": [
        [
          75.02,
          74.36,
          72.39999999999999,
          72.08
        ]
      ],
      "250": [
        [
          84.39999999999999,
          81.82000000000001
        ]
      ]
    },
    "MarginSamplingSelection": {
      "10": [
        [
          25.72,
          38.83,
          53.99,
          57.53,
          62.39,
          66.58,
          64.23,
          67.5,
          72.75,
          76.35,
          77.31,
          77.96,
          78.17,
          80.67,
          81.76,
          83.63000000000001,
          85.26,
          85.37,
          85.82,
          85.76,
          86.00999999999999,
          87.01,
          87.45,
          87.83,
          88.64999999999999,
          88.1,
          87.92999999999999,
          89.21,
          89.57000000000001,
          90.34,
          91.12,
          90.86,
          91.17,
          91.53999999999999,
          91.84,
          91.75999999999999,
          91.07,
          91.36,
          91.22,
          91.88,
          92.0,
          91.97,
          91.7,
          92.12,
          92.28,
          92.57,
          92.63,
          92.55,
          92.80000000000001,
          92.75999999999999
        ]
      ],
      "25": [
        [
          39.050000000000004,
          51.85999999999999,
          67.30000000000001,
          74.29,
          74.42999999999999,
          80.15,
          83.55,
          83.58,
          86.24000000000001,
          87.22999999999999,
          87.89,
          88.42999999999999,
          88.94,
          90.06,
          89.57000000000001,
          90.03999999999999,
          90.03999999999999,
          91.14,
          91.86,
          91.79
        ]
      ],
      "50": [
        [
          62.99,
          75.07000000000001,
          75.38,
          83.96000000000001,
          87.89,
          89.62,
          89.7,
          91.47999999999999,
          91.84,
          92.49000000000001
        ]
      ],
      "125": [
        [
          76.4,
          85.66,
          90.22,
          92.13
        ]
      ],
      "250": [
        [
          81.26,
          87.98
        ]
      ]
    },
    "RandomSelection": {
      "10": [
        [
          36.01,
          43.34,
          53.94,
          58.02,
          60.97,
          66.44,
          68.63,
          70.8,
          72.08,
          73.58,
          73.63,
          74.0,
          74.2,
          75.52,
          75.41,
          75.81,
          75.98,
          77.97,
          77.5,
          80.15,
          79.81,
          81.13,
          80.87,
          80.58999999999999,
          80.86,
          81.26,
          82.26,
          82.44,
          82.75,
          82.92,
          83.76,
          83.50999999999999,
          85.05,
          85.50999999999999,
          85.13,
          86.02,
          86.02,
          86.5,
          86.09,
          85.83,
          86.9,
          86.46000000000001,
          86.38,
          86.88,
          87.09,
          87.87,
          87.47,
          87.59,
          87.74,
          87.78
        ]
      ],
      "25": [
        [
          51.64,
          60.72,
          65.45,
          70.28999999999999,
          72.94,
          76.08,
          77.51,
          77.78,
          79.35,
          80.39,
          81.6,
          81.17,
          82.73,
          84.28,
          84.15,
          85.2,
          86.13,
          86.78,
          86.95,
          87.59
        ]
      ],
      "50": [
        [
          56.86,
          69.43,
          71.87,
          75.68,
          80.01,
          82.06,
          84.5,
          85.92,
          86.76,
          87.32
        ]
      ],
      "125": [
        [
          71.5,
          82.39,
          85.76,
          87.56
        ]
      ],
      "250": [
        [
          83.19,
          88.14999999999999
        ]
      ]
    }
  },
  "SvmModel": {
    "EntropySelection": {
      "10": [
        [
          35.93,
          35.97,
          40.33,
          39.39,
          41.349999999999994,
          42.99,
          46.23,
          46.18,
          47.260000000000005,
          52.5,
          52.400000000000006,
          51.25999999999999,
          51.690000000000005,
          51.800000000000004,
          53.18000000000001,
          53.82,
          55.88999999999999,
          56.120000000000005,
          57.269999999999996,
          59.41,
          59.95,
          62.629999999999995,
          61.339999999999996,
          63.88,
          65.34,
          65.77,
          66.9,
          67.96,
          68.27,
          67.44,
          68.45,
          68.63,
          68.0,
          68.47,
          68.77,
          68.8,
          69.17,
          68.97,
          69.33,
          69.67999999999999,
          69.95,
          70.34,
          70.47,
          71.19,
          71.97,
          72.26,
          72.06,
          71.98,
          72.55,
          72.78999999999999
        ]
      ],
      "25": [
        [
          50.33,
          51.59,
          56.599999999999994,
          60.24,
          61.57,
          63.5,
          66.74,
          68.19,
          68.02,
          69.8,
          75.88000000000001,
          77.24,
          78.09,
          79.38,
          80.4,
          80.99,
          80.28999999999999,
          80.12,
          79.75999999999999,
          80.36999999999999
        ]
      ],
      "50": [
        [
          64.55,
          68.75,
          71.34,
          74.11999999999999,
          75.96000000000001,
          77.03999999999999,
          76.85,
          79.19,
          80.51,
          80.99
        ]
      ],
      "125": [
        [
          78.4,
          78.21000000000001,
          80.08,
          81.0
        ]
      ],
      "250": [
        [
          83.11,
          84.53
        ]
      ]
    },
    "MarginSamplingSelection": {
      "10": [
        [
          31.86,
          37.6,
          46.23,
          58.41,
          60.34,
          65.98,
          65.01,
          69.86,
          72.28999999999999,
          74.65,
          76.24,
          77.37,
          77.59,
          78.7,
          79.06,
          80.06,
          81.21000000000001,
          82.32000000000001,
          83.17,
          83.78,
          84.6,
          84.58,
          84.47,
          85.07000000000001,
          85.64,
          85.75,
          85.7,
          86.18,
          86.58,
          86.50999999999999,
          86.71,
          87.09,
          86.91,
          87.13,
          87.42999999999999,
          87.75,
          87.97,
          88.23,
          88.23,
          88.22,
          88.16000000000001,
          88.03,
          88.38000000000001,
          88.39,
          88.64,
          88.98,
          89.05999999999999,
          89.34,
          89.3,
          89.52
        ]
      ],
      "25": [
        [
          50.160000000000004,
          63.73,
          70.14,
          74.4,
          78.99000000000001,
          80.36999999999999,
          82.76,
          84.35000000000001,
          85.78,
          86.61999999999999,
          87.46000000000001,
          87.64,
          88.08,
          88.75,
          88.68,
          89.02,
          89.39,
          89.57000000000001,
          89.66,
          89.8
        ]
      ],
      "50": [
        [
          59.099999999999994,
          69.91000000000001,
          75.66000000000001,
          81.41000000000001,
          82.69,
          85.07000000000001,
          85.92,
          86.89,
          87.8,
          87.6
        ]
      ],
      "125": [
        [
          79.45,
          84.1,
          86.36,
          88.16000000000001
        ]
      ],
      "250": [
        [
          84.58,
          87.38
        ]
      ]
    },
    "RandomSelection": {
      "10": [
        [
          31.009999999999998,
          33.54,
          48.67,
          57.34,
          61.68,
          64.75999999999999,
          69.28,
          71.97,
          73.2,
          74.03,
          75.08,
          76.01,
          76.55,
          77.42999999999999,
          77.75,
          79.72,
          80.78,
          81.62,
          81.8,
          82.03,
          81.82000000000001,
          82.72,
          82.94,
          83.03,
          83.72,
          83.81,
          83.72,
          83.95,
          84.38,
          85.02,
          85.36,
          85.65,
          85.26,
          85.37,
          85.37,
          85.19,
          85.18,
          85.61999999999999,
          85.47,
          85.61999999999999,
          85.78,
          86.00999999999999,
          86.05000000000001,
          85.96000000000001,
          85.96000000000001,
          86.0,
          86.24000000000001,
          86.6,
          86.66,
          86.81
        ]
      ],
      "25": [
        [
          51.449999999999996,
          67.88,
          72.25,
          77.82,
          79.19,
          80.32000000000001,
          81.19,
          81.91000000000001,
          82.67999999999999,
          82.89,
          82.89,
          83.05,
          83.91999999999999,
          84.69,
          84.77,
          84.66,
          85.0,
          85.3,
          85.79,
          86.08
        ]
      ],
      "50": [
        [
          56.38999999999999,
          72.92999999999999,
          78.44,
          81.0,
          82.73,
          83.52000000000001,
          84.35000000000001,
          85.17,
          86.17,
          86.66
        ]
      ],
      "125": [
        [
          73.76,
          83.52000000000001,
          85.6,
          86.44
        ]
      ],
      "250": [
        [
          83.02000000000001,
          85.64
        ]
      ]
    }
  }
}

