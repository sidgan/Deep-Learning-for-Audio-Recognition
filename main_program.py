#!usr/local/bin 

from pylab import * 
from scipy.io import wavfile 
import pylab 
import scipy 
import os 
import numpy 
import sys
sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages') 
import sklearn

from scipy.io import wavfile

from sklearn.kernel_approximation import RBFSampler
import sklearn.cluster 
import optparse
#import copper 


from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier


from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation 
from sklearn.cross_validation import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm 
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
import random
import csv as csv
import pandas as pd 
import numpy as np 
import warnings 
warnings.simplefilter('ignore', DeprecationWarning)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline 
import matplotlib.pyplot as plt 
import sklearn
import Image 
from sklearn.ensemble import AdaBoostClassifier
import pylab as pl 
from PIL import Image 
import os 

#sampling_rate, numpy_array= scipy.io.wavfile.read('train1.wav')
#print sampling_rate
#print numpy_array


#setup a standard image size; this will distort some images but will get everything into the same shape
STANDARD_SIZE = (300, 167)
def img_to_matrix(filename, verbose=False):
	"""
	takes a filename and turns it into a numpy array of RGB pixels	
	"""
	img = Image.open(filename)
	if verbose==True:
		print "changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE))
	img = img.resize(STANDARD_SIZE)
	img = list(img.getdata())
	img = map(list, img)
	img = np.array(img)
	return img
 
def flatten_image(img):
	"""
	takes in an (m, n) numpy array and flattens it
	into an array of shape (1, m * n)
	"""
	s = img.shape[0] * img.shape[1]
	img_wide = img.reshape(1, s)
	return img_wide[0] 

#LOCATION OF DATA AND IMPORT 
#DATA-----------------------------------
img_dir = "/home/sidgan/Downloads/MAJOR/images/"
images = [img_dir+ f for f in os.listdir(img_dir)]
labels = ["human" if "h" in f.split('/')[-1] else "not_human" for f in images]
 
data = []
for image in images:
	img = img_to_matrix(image)
	img = flatten_image(img)
	data.append(img)
 
data = np.array(data)
#print data 

is_train = np.random.uniform(0, 1, len(data)) <= 0.7
y = np.where(np.array(labels)=="human", 1, 0)

train_x, train_y = data[is_train], y[is_train]
test_x, test_y = data[is_train==False], y[is_train==False]



pca = RandomizedPCA(n_components=2)
X = pca.fit_transform(data)
df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "label":np.where(y==1, "HUMAN", "NOT HUMAN")})
colors = ["red", "yellow"]
for label, color in zip(df['label'].unique(), colors):
	mask = df['label']==label
	pl.scatter(df[mask]['x'], df[mask]['y'], c=color, label=label)
pl.legend()
pl.show() 


pca = RandomizedPCA(n_components=5)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x) 


#print train_x[:5]

#PERFORMS CROSS VALDIATION 	
def cal_score(method, clf, features_test, target_test):
		scores = cross_val_score(clf, features_test, target_test)
		print method + " : %f " % scores.max()
		#print scores.max()		



''' 
 
data = []
for image in images:
	img = img_to_matrix(image)
	img = flatten_image(img)
	data.append(img)



'''
# Test the classifier

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(train_x, train_y) 
###############################
#PREPROCESSING OF SAMPLE DATA SET 
'''
img_dir = "/home/sidgan/Downloads/MAJOR/images/"
images = [img_dir+ f for f in os.listdir(img_dir)]
labels = ["human" if "h" in f.split('/')[-1] else "not_human" for f in images]
 
data = []
for image in images:
	img = img_to_matrix(image)
	img = flatten_image(img)
	data.append(img)
 
data = np.array(data)
#print data 

is_train = np.random.uniform(0, 1, len(data)) <= 0.7
y = np.where(np.array(labels)=="human", 1, 0)

'''
sample_img_dir = "/home/sidgan/Downloads/MAJOR/New_folder/sample/"
sample_images = [sample_img_dir+ f for f in os.listdir(sample_img_dir)]
sample_labels = ["human" if "h" in f.split('/')[-1] else "not_human" for f in sample_images]
sample_data = []
for sample_image in sample_images:
	img = img_to_matrix(sample_image)
	img = flatten_image(img)
	sample_data.append(img)
 
sample_data = np.array(sample_data)
#print sample_data
sample_is_train = np.random.uniform(0, 1, len(sample_data)) <= 0.7
sample_y = np.where(np.array(sample_labels)=="human", 1, 0)


'''

train_x, train_y = data[is_train], y[is_train]
test_x, test_y = data[is_train==False], y[is_train==False]



#pca = RandomizedPCA(n_components=2)
#X = pca.fit_transform(data)
#df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "label":np.where(y==1, "HUMAN", "NOT HUMAN")})
#colors = ["red", "yellow"]
#for label, color in zip(df['label'].unique(), colors):
#	mask = df['label']==label
#	pl.scatter(df[mask]['x'], df[mask]['y'], c=color, label=label)
#pl.legend()
#pl.show() 


pca = RandomizedPCA(n_components=5)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x) 

'''

#print sample_y
#print knn.predict(sample_data)
#scores = cross_val_score(knn, sample_y, sample_data)
#print scores.max()
'''
>>> X = [[0], [1], [2], [3]]
>>> y = [0, 0, 1, 1]
>>> from sklearn.neighbors import KNeighborsRegressor
>>> neigh = KNeighborsRegressor(n_neighbors=2)
>>> neigh.fit(X, y) 
KNeighborsRegressor(...)
>>> print(neigh.predict([[1.5]]))
[ 0.5]
'''


#from sklearn.neighbors import KNeighborsRegressor
#neigh = KNeighborsRegressor(n_neighbors=2)
#neigh.fit(train_x, train_y) 
#print(neigh.predict([[sample_data]]))


#################
print pd.crosstab(test_y, knn.predict(test_x), rownames=["Actual"], colnames =["Predicted"])
#print pd.crosstab(sample_y, knn.predict(sample_data), rownames=["ACTUAL"], colnames=["PREDICTED"])

clf_ada = AdaBoostClassifier(n_estimators=100)
params = {
			'learning_rate': [.05, .1,.2,.3,2,3, 5],
			'max_features': [.25,.50,.75,1],
			'max_depth': [3,4,5],
			}
gs = GridSearchCV(clf_ada, params, cv=5, scoring ='accuracy', n_jobs=4)
clf_ada.fit(train_x, train_y)
cal_score("ADABOOST",clf_ada, test_x, test_y)

features_test = test_x 
target_test = test_y 
features_train = train_x 
target_train = train_y

prob = 1
#Naive Bayes 
nb_estimator = GaussianNB()
nb_estimator.fit(features_train, target_train)
cal_score("NAIVE BAYES CLASSIFICATION",nb_estimator, features_test, target_test)
#predictions = nb_estimator.predict(test)
#SVC Ensemble

rf = RandomForestClassifier(n_estimators=100)
rf = rf.fit(features_train, target_train)
cal_score("RANDOM FOREST CLASSIFIER",rf, features_test, target_test)
#predictions = rf.predict_proba(test)

#Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, subsample=.8)
params = {
		'learning_rate': [.05, .1,.2,.3,2,3, 5],
		'max_features': [.25,.50,.75,1],
		'max_depth': [3,4,5],
		}
gs = GridSearchCV(gb, params, cv=5, scoring ='accuracy', n_jobs=4)
gs.fit(features_train, target_train)
cal_score("GRADIENT BOOSTING",gs, features_test, target_test)
		#sorted(gs.grid_scores_, key = lambda x: x.mean_validation_score)
		#print gs.best_score_
		#print gs.best_params_
		#predictions = gs.predict_proba(test)
		#KERNEL APPROXIMATIONS - RBF 		
rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(data)
		
#SGD CLASSIFIER		
clf = SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
      		fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
       loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5,
       random_state=None, shuffle=True, verbose=0,
       warm_start=False)
clf.fit(features_train, target_train)
cal_score("SGD Regression",clf, features_test, target_test)

#KN Classifier
neigh = KNeighborsClassifier(n_neighbors = 1)
neigh.fit(features_train, target_train)
cal_score("KN CLASSIFICATION",neigh, features_test, target_test)

clf_tree = tree.DecisionTreeClassifier(max_depth=10)
clf_tree.fit(features_train, target_train)
cal_score("DECISION TREE CLASSIFIER",clf_tree, features_test, target_test)
	
#LOGISTIC REGRESSION 
logreg = LogisticRegression(C=3)
logreg.fit(features_train, target_train)
cal_score("LOGISTIC REGRESSION",logreg, features_test, target_test)
#predictions = logreg.predict(test)
# SUPPORT VECTOR MACHINES 
clf = svm.SVC(kernel = 'linear')
clf.fit(features_train, target_train)
cal_score("LINEAR KERNEL",clf, features_test, target_test)
#print clf.kernel
#for sigmoid kernel
clf= svm.SVC(kernel='rbf', C=2).fit(features_train, target_train)
cal_score("SVM RBF KERNEL",clf, features_test, target_test)		
#predictions = clf.predict(test)
#Lasso 
clf = linear_model.Lasso(alpha=.1)
clf.fit(features_train, target_train)
cal_score("LASSO",clf, features_test, target_test)
#elastic net 
clf = linear_model.ElasticNet(alpha=.1, l1_ratio=.5, fit_intercept=True, normalize=False, precompute='auto',max_iter=1000, copy_X=True, tol =.0001, warm_start=False, positive=False)
clf.fit(features_train, target_train)
cal_score("ELASTIC NET",clf, features_test, target_test)
#SGD REGRESSION	
clf = SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0,
     		fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
       loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5,
       random_state=None, shuffle=True, verbose=0,
       warm_start=False)
clf.fit(features_train, target_train)
cal_score("SGD Regression",clf, features_test, target_test)

prob = 3
#MINI BATCH K MEANS CLUSTERING
clf = sklearn.cluster.MiniBatchKMeans(init='k-means++', max_iter=100, batch_size=100, verbose=0, compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01)
clf.fit(features_train, target_train)

		#MEAN SHIFT
			
clf = sklearn.cluster.MeanShift(bandwidth=None, seeds=[features_train, target_train], bin_seeding=False, min_bin_freq=1, cluster_all=True)
		#clf.fit([features_train, target_train])
		#clf.fit(data, target)		
		#if options.cross_validation == 'True': 
		#	cal_score("MEAN SHIFT CLUSTERING",clf, features_test, target_test)
		#K MEANS CLUSTERING	
clf = sklearn.cluster.KMeans( init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances=True, verbose=0, random_state=None, copy_x=True, n_jobs=1)
clf.fit(data)
		#if options.cross_validation == 'True': 
		#	cal_score("K MEANS CLUSTERING",clf, features_test, target_test)


prob = 4
#PCA
pca = PCA(n_components=1)
pca_train = pca.fit(data)
pca_test = pca.transform(test)


