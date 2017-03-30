###
#Writer:  Koyo Miyamura
#Summary: Machine learning script using performance counter (CSV file).
#Input:   Performance counter from each programs and labels
#         (Input csv is got using program at https://github.com/koyo-miyamura/perf_analyze_rewrite)
#Output:  Accuracy rate of all input
#         Classification_report
#         Confusion matrix
#         Accuracy after moving average
#         Plot of decision function
#Remarks: I labeled CHANGE in the place which I thought you want to change. (you can search the word Chan to find the variable places)
###

from time import time
import numpy as np
import csv
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC,SVC
from sklearn import preprocessing, neighbors
from sklearn.externals import joblib

###############################################################################
# Download from csv
#CHANGE
bzip2_train  = np.loadtxt("csv/bzip2_train.csv", delimiter = ";", skiprows=1) 
sphinx3_train= np.loadtxt("csv/sphinx3_train.csv", delimiter = ";", skiprows=1)
bzip2_test   = np.loadtxt("csv/bzip2_test.csv", delimiter = ";", skiprows=1)
sphinx3_test = np.loadtxt("csv/sphinx3_test.csv", delimiter = ";", skiprows=1) 
bzip2_2      = np.loadtxt("csv/bzip2_2_test.csv", delimiter = ";", skiprows=1)
gcc          = np.loadtxt("csv/gcc.csv", delimiter = ";", skiprows=1)
perlbench    = np.loadtxt("csv/perlbench.csv", delimiter = ";", skiprows=1)
mcf          = np.loadtxt("csv/mcf.csv", delimiter = ";", skiprows=1)
libquantum   = np.loadtxt("csv/libquantum.csv", delimiter = ";", skiprows=1)
bwaves       = np.loadtxt("csv/bwaves.csv", delimiter = ";", skiprows=1)
lbm          = np.loadtxt("csv/lbm.csv", delimiter = ";", skiprows=1)
wrf          = np.loadtxt("csv/wrf.csv", delimiter = ";", skiprows=1)
hmmer        = np.loadtxt("csv/hmmer.csv", delimiter = ";", skiprows=1)

# Set the data and label
#CHANGE
X_train = np.concatenate([bwaves[:,1:-1], sphinx3_train[:,1:-1], mcf[:,1:-1], gcc[:,1:-1], perlbench[:,1:-1], bzip2_test[:,1:-1], wrf[:,1:-1] ])

y_train = np.concatenate([map(int,bwaves[:,-1]), map(int,sphinx3_train[:,-1]), map(int,mcf[:,-1]), map(int,gcc[:,-1]), map(int,perlbench[:,-1]), map(int,bzip2_test[:,-1]), map(int,wrf[:,-1])])

target_names = np.array(['others','bwaves'])

X_test = np.concatenate([bwaves[:,1:-1], hmmer[:,1:-1], libquantum[:,1:-1], lbm[:,1:-1] ])

y_test = np.concatenate([map(int,bwaves[:,-1]), map(int,hmmer[:,-1]), map(int,libquantum[:,-1]), map(int,lbm[:,-1])])

# Parameter of data
n_classes       = target_names.shape[0]
n_samples_train = X_train.shape[0]
n_features_train= X_train.shape[1]
n_samples_test  = X_test.shape[0]
n_features_test = X_test.shape[1]

# Check 
print("Total dataset size:")
print("n_samples_train : %d" % n_samples_train )
print("n_samples_test  : %d" % n_samples_test  )
print("n_features_train: %d" % n_features_train)
print("n_features_test : %d" % n_features_test )
print("n_classes: %d" % n_classes)

'''
###############################################################################
# Compute a PCA on the dataset (if needed)
#CHANGE
n_components = 5

print("Extracting the top %d eigendatas from %d data"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, 
          #svd_solver='randomized',
          #whiten=True
          ).fit(X_train)

# Check
print("done in %0.3fs" % (time() - t0))
#print("pca.components shape")
#print(pca.components_.shape)

#print(pca.components_)
#print(pca.explained_variance_) 
#print(pca.explained_variance_ratio_) 
#E = pca.explained_variance_ratio_ 
#print(E)
#print "cumsum E", np.cumsum(E) 

X_train = pca.transform(X_train)
X_test  = pca.transform(X_test)

#print(" X_train_pca.shape,X_test_pca.shape")
#print(X_train_pca.shape,X_test_pca.shape)

"""
# Check
print(X_test.mean(axis=0),X_test_pca.mean(axis=0))
print(X_train.mean(axis=0),X_train_pca.mean(axis=0))
print(X_test.std(axis=0),X_test_pca.std(axis=0))
print(X_train.std(axis=0),X_train_pca.std(axis=0))
"""

'''
###############################################################################
# Preprocessing data (if needed) 

#Scale the deta (0,1)
min_max_scaler = preprocessing.MinMaxScaler()
#Memorize mean and std of the train data
X_train = min_max_scaler.fit_transform(X_train)
#Transform test data using train data's mean and std
X_test          = min_max_scaler.transform(X_test)

#CHANGE
bzip2_trans     = min_max_scaler.transform(bzip2_test[:,1:-1]) 
bzip2_2_trans   = min_max_scaler.transform(bzip2_2[:,1:-1])
hmmer_trans     = min_max_scaler.transform(hmmer[:,1:-1]) 
libq_trans      = min_max_scaler.transform(libquantum[:,1:-1]) 
bwaves_trans    = min_max_scaler.transform(bwaves[:,1:-1]) 
lbm_trans       = min_max_scaler.transform(lbm[:,1:-1]) 
wrf_trans       = min_max_scaler.transform(wrf[:,1:-1]) 

#Scale the data mean=0,std=1
scaler          = preprocessing.StandardScaler().fit(X_train)
X_train         = scaler.transform(X_train)
X_test          = scaler.transform(X_test)

#CHANGE
bzip2_trans     = scaler.transform(bzip2_trans)
bzip2_2_trans   = scaler.transform(bzip2_2_trans)
hmmer_trans     = scaler.transform(hmmer_trans)
libq_trans      = scaler.transform(libq_trans)
bwaves_trans    = scaler.transform(bwaves_trans)
lbm_trans       = scaler.transform(lbm_trans)
wrf_trans       = scaler.transform(wrf_trans)

#Check 
print("X_train_pca whitened std")
print(X_train.std(axis=0))
print("X_test_pca whitened std")
print(X_test.std(axis=0))
print("X_train_pca whitened mean")
print(X_train.mean(axis=0))
print("X_test_pca whitened mean")
print(X_test.mean(axis=0))

###############################################################################
# Train a SVM classification model
#CHANGE
print("Fitting the classifier to the training set")
t0 = time()
#param_grid = {'C': [1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5],'kernel': ['rbf','sigmoid','poly']}
#param_grid = {'C': [1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5]}
#clf = GridSearchCV(SVC(), param_grid)
clf = SVC(C=1,kernel='poly')
clf = clf.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
#print("Best estimator found by grid search:")
#print(clf.best_estimator_)

#CHANGE
decision        = clf.decision_function(X_test)
decision_bzip   = clf.decision_function(bzip2_trans)
decision_bzip_2 = clf.decision_function(bzip2_2_trans)
decision_hmmer  = clf.decision_function(hmmer_trans)
decision_libq   = clf.decision_function(libq_trans)
decision_bwaves = clf.decision_function(bwaves_trans)
decision_lbm    = clf.decision_function(lbm_trans)
decision_wrf    = clf.decision_function(wrf_trans)

'''
# Other Algorithm
# Train a KNN classification model
t0 = time()
#param_grid = {'n_neighbors': [i+1 for i in range(100)]}
param_grid = {'n_neighbors': [1,2,3,4,5,10,15]}
clf = GridSearchCV(neighbors.KNeighborsClassifier(), param_grid)
clf = clf.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
'''

###############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting the test set")
t0 = time()
y_pred = clf.predict(X_test)
print("done in %0.3fs" % (time() - t0))

print("---------------Result----------------")
#print("decision_function")
#print(decision)

print("accuracy = %f" % (accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred,target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
###############################################################################
# Plot the result

# Each decision function plot (moving average)
# CHANGE
list = [decision_bwaves, decision_hmmer, decision_libq, decision_lbm]
listname = ["bwaves","hmmer","libquantum","lbm"]
window = 10
b = np.ones(window)/float(window)
for i in range(len(list)):
    func = list[i]
    name = listname[i]
    decision_avg = np.convolve(func,b,'same')
    x = np.array([(i+1)*0.1 for i in range(len(func))])
    plt.plot(x,decision_avg, marker="o",linestyle="none",markersize=2, label = name + " window=" + str(window), color="blue")
    plt.xlabel('Time [sec]')
    plt.ylabel('Distance from hyperplane')
    plt.legend()
    plt.show()

"""
# Moving Average accuracy 
threshold = 0
#plt.hlines(y=threshold, xmin=0, xmax=len(func)*0.1, colors='r', linewidths=2,label="threshold")
result = []
for i in range(len(decision_avg)):
    if decision_avg[i]>=threshold:
        result.append(1)
    else:
        result.append(0)
print("moving_averaged accuracy = %f" % (accuracy_score(lbm[: -1],np.array(result))))
"""

# Plot all decision functions
x = np.array([(i+1)*0.1 for i in range(n_samples_test)])
plt.plot(x,decision, marker="o",linestyle="none",markersize=3, label = "decision function", color="blue")

plt.xlabel('Time [sec]')
plt.ylabel('Distance from hyperplane')
plt.legend()
#plt.ylim([-5,max(decision_libq)])
plt.show()

"""
# Save the result (if you use this, you cannot use GridSearchCV. So after you use GridSearch and know the best parameters, you should use it.)
cname ='clf.pkl'
print("save clf as %s",cname)
joblib.dump(clf ,cname) 
tname ='trans.pkl' 
print("save min_max_scaler as %s",tname)
joblib.dump(min_max_scaler ,tname)
tname ='scalar.pkl' 
print("save scaler as %s",tname)
joblib.dump(scaler ,tname)
"""
