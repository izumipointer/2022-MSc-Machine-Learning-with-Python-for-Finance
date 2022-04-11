
#########SET WORKING DIRECTORY##################################################################
# import the packages we use
import os
# set the working directory:
os.chdir('C:/Users/agarel/OneDrive - Audencia Business School/Documents/machine learning and trading')
# print the current working directory to make sure it is the right one:
print("Current working directory: {0}".format(os.getcwd()))
################################################################################################

################################################################################################
###GET TO KNOW THE DATA, CLASSES AND FEATURES###################################################
################################################################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#console display settings for pandas dataframes
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# import the csv data:
dataset=pd.read_csv('iris_flowers_data.csv', index_col=0)
# show information about the variables:
dataset.info()
# show a snapshot of the data:
dataset.head()
# how many instances for each species:
dataset.groupby('class').size()
# We show the mean value by pair of feature-specie
dataset.groupby('class').mean()
# what is the correlation between the features?
dataset.corr()
# we can show it using a scatter plot matrix
plt.clf()
pd.plotting.scatter_matrix(dataset)
plt.show()
# do some of the correlations between features vary from one specie to another?
dataset.groupby('class').corr()
# we can show the same information but visually with a pairplot, we use the seaborn package
# this plot also provides additional information on the distribution of features across species
plt.clf()
sns.set_palette('husl')
sns.pairplot(dataset, hue='class', markers='+')
plt.show()
# a violin plot shows the distribution of values per feature for each specie
# each violin shows lines for the median value, the top quartile value, and the bottom quartile value
plt.clf()
sns.violinplot(y='class', x='sepal-length', data=dataset, inner='quartile')
plt.show()
sns.violinplot(y='class', x='sepal-width', data=dataset, inner='quartile')
plt.show()
sns.violinplot(y='class', x='petal-length', data=dataset, inner='quartile')
plt.show()
sns.violinplot(y='class', x='petal-width', data=dataset, inner='quartile')
plt.show()
################################################################################################

################################################################################################
###MACHINE LEARNING WITH SKLEARN################################################################
################################################################################################
#additional imports
from sklearn.model_selection import train_test_split

##CREATE TRAIN AND SAMPLE TEST###################################################################
# we create a dataframe of features only
X = dataset.drop(['class'], axis=1)
print(X.head())
print(X.shape)
# we create a dataframe of what we want to predict: the iris species (class)
y = dataset['class']
print(y.head())
print(y.shape)
# we create a train and test samples for X and y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
print("Size of the train sample of features is :", X_train.shape)
print("Size of the train sample of species (labels) is :", y_train.shape)
print("Size of the test sample of features is :", X_test.shape)
print("Size of the test sample of species (labels) is :", y_test.shape)
# show train sample features data
print(X_train.head())
# show train sample class data
print(y_train.head())
# show test sample features data
print(X_test.head())
# show test sample class data
print(y_test.head())
################################################################################################

##K-NEAREST NEIGHBORS#############################################################################
# we have clusters in the train sample:
full_train_sample = pd.concat([X_train, y_train], axis=1)
sns.FacetGrid(full_train_sample, hue ="class",
              height = 6).map(plt.scatter,
                              'sepal-length',
                              'petal-length').add_legend()    

# create a function that calculates the Euclidean distance between two vectors
from math import sqrt
def euclidean_distance(v1, v2):
    distance = 0.0
    for i in range(len(v1)-1):
        distance += (v1[i] - v2[i])**2
    return sqrt(distance)                              
                              
# calculate the Euclidean distance between the values for the four iris features of the observation we want to classify
# we want to classify the first flower (observation) of the test sample
features_obs_0 = X_test.iloc[0, :]
print(features_obs_0)                              
                              
# we create a distance matrix that stores the Euclidean distances between the first observation of the test sample
# and all the observations of the train sample
distance_matrix=full_train_sample.reset_index()
distance_matrix['distance']=99.99
print(distance_matrix.head())
print("Matrix shape is :", distance_matrix.shape)   

# get rid off warning messages
pd.options.mode.chained_assignment = None
# we compute the Euclidian distances by calling the function we created
for row in range(0,y_train.count()):
    distance = euclidean_distance(features_obs_0, X_train.iloc[row, :])
    distance_matrix['distance'][row]=distance
# we show the updated matrix with computed distances
print(distance_matrix.head())        

# then we look for the closest observations 
# that is the flowers which have the lowest Euclidean distances based on their features values
print(distance_matrix[["class","distance"]].sort_values(by = 'distance').head(10))


# we redo the same prediction but using the sklearn package
# we import the sklearn package for this type of classifier
from sklearn.neighbors import KNeighborsClassifier
# we specify the number of closest neighbors we want to consider, 3 here
knn = KNeighborsClassifier(n_neighbors=3)
# we fit the model to the training data
knn.fit(X_train, y_train)
# we predict values based on the model parameters
y_predict=knn.predict(X_test)
# we show the prediction for the first observation of the test sample
print(y_predict[0])
# of course we made predictions for all the observations fo the test sample here
print(y_predict)


# ACCURACY OF THE PREDICTION
import numpy as np
# a first thing we can do, manually, is to list the actual species of the test sample next to the predicted species
predict_and_actual = pd.concat([y_test.reset_index(),pd.DataFrame(y_predict)],axis=1)
predict_and_actual = predict_and_actual.rename(columns={'class': 'Actual Species', 0 : 'Predicted Species'})
# we can create a flag for good predictions
predict_and_actual['Accurate Prediction']=np.where(predict_and_actual['Actual Species']==predict_and_actual['Predicted Species'], True, False)
print(predict_and_actual)
#finally we can report the percentage of True (accurate perdictions)
print(predict_and_actual['Accurate Prediction'].value_counts(normalize=True))

# we can also rely on function of the sklearn package
# score returns the % of good predictions
knn.score(X_test,y_test) 

#confusion matrix
from sklearn import metrics
result = metrics.confusion_matrix(y_test, y_predict)
cmd = metrics.ConfusionMatrixDisplay(result, display_labels=['Setosa', 'Versicolor','Virginica'])
print("Confusion Matrix:")
print(result)
cmd.plot()

#classification report
print(metrics.classification_report(y_test, y_predict))

#select best KNN model
# we create k_range: a list of integer values from 1 to 20
k_range = list(range(1,20))
# we create an empty list to store the accuracy scores of each model
scores = []
# we use 20 different KNN classifier, with each time a different value for the parameter n_neighbors (k)
# we store the associated accuracy score (macro avg.) in the score list
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    #get fraction of correct predictions
    scores.append(metrics.accuracy_score(y_test, y_pred))
# we show a plot of the accuracy score by value of the parameter k (how many neighbors we consider)
plt.clf()
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores of KNN for values of k')
plt.show()

#conclusion based on 10 folds
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# create ten folds
cv = KFold(n_splits=10, random_state=1, shuffle=True)
# create classifier - we pick the best one based on previous analysis
model = KNeighborsClassifier(n_neighbors=1)
# generate the accuracy scores of the classifier for each fold
# we use the fraction of true positive
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv)
print(scores)
# report performance - mean accuracy score over the 10 folds and its standard deviation
print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


#finding best classifier with 10 folds
k_range = list(range(1,20))
scores_list = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv = KFold(n_splits=10, random_state=k, shuffle=True)
    scores = cross_val_score(knn, X, y, scoring='accuracy', cv=cv)
    scores_list.append(np.mean(scores))
plt.clf()
plt.plot(k_range, scores_list)
plt.xlabel('Value of k for KNN')
plt.ylabel('Mean Accuracy Score over 10 folds')
plt.title('Accuracy Scores of KNN for values of k based on 10 folds')
plt.show()
# for each k show the average accuracy score for 10 folds
print(pd.DataFrame(scores_list,k_range,columns =['Accuray Score']))


##DECISION TREE#############################################################################
#calculation of information gain to choose split point
full_train_sample = pd.concat([X_train, y_train], axis=1).reset_index()

# function that computes the gini index for the first node (the parent node)
def gini_index_parent():
   full= full_train_sample["class"].count()
   found= full_train_sample.groupby('class').count()
   result=(found/full)['index'].fillna(0)
   return 1 - (result**2).sum()
# test the function
gini_index_parent()

# function that computes the Gini Index for the first child node
def gini_index_node_1(feature, spliting_point):
   full= full_train_sample[full_train_sample[feature] <= spliting_point].count()
   found=full_train_sample[full_train_sample[feature] <= spliting_point].groupby('class').count()
   result=(found/full)['index'].fillna(0)
   return 1 - (result**2).sum()
# test the function
gini_index_node_1("petal-length", 2.45)

# function that computes the Gini Index for the other child node
def gini_index_node_2(feature, spliting_point):
   full= full_train_sample[full_train_sample[feature] > spliting_point].count()
   found=full_train_sample[full_train_sample[feature] > spliting_point].groupby('class').count()
   result=(found/full)['index'].fillna(0)
   return 1 - (result**2).sum()
# test the function
gini_index_node_2("petal-length", 2.45)

# function that computes the information gain
def information_gain (feature, spliting_point):
   full= len(full_train_sample)
   w1=full_train_sample[full_train_sample[feature] <= spliting_point].index.size / full
   w2=full_train_sample[full_train_sample[feature] > spliting_point].index.size / full
   return gini_index_parent() - w1 * gini_index_node_1(feature,spliting_point) - w2 * gini_index_node_2(feature,spliting_point)
# test the function
information_gain("petal-length", 2.45)

# function that tries all possible spliting points for a given feature based on its max and min values in the train sample
def find_optimal_spliting_point(feature, minv, maxv):
    for i in range(minv,maxv):
        print(feature, i/10, information_gain(feature,i/10))
# find optimal split point for petal-length
find_optimal_spliting_point("petal-length", 10, 69)
#  find optimal split point for petal-length
find_optimal_spliting_point("petal-width", 1, 25)

# we know use the package of sklearn to get the same result
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
#defautl criterion is gini
clf = DecisionTreeClassifier(max_depth = 1, random_state = 0)
clf.fit(X_train, y_train)
#we show tree information
tree.plot_tree(clf, feature_names=['sepal-length','sepal-width', 'petal-length', 'petal-width'],  class_names=['setosa', 'versicolor', 'class3'], filled=True)

# decision tree with a depth of 2
clf = DecisionTreeClassifier(max_depth = 2, random_state = 0)
clf.fit(X_train, y_train)
# we show tree information
tree.plot_tree(clf, feature_names=['sepal-length','sepal-width', 'petal-length', 'petal-width'],  class_names=['setosa', 'versicolor', 'class3'], filled=True)

# importance of the features in the decision tree
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(clf.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False)
print(importances)

# Predict the species based on the decision tree with a depth of 2
clf.predict(X_test)
# The score method returns the accuracy of the model
score = clf.score(X_test, y_test)
print(score)

#we pick the depth that maximizes the accuracy of the decision tree:
# List of values to try for max_depth:
max_depth_range = list(range(1, 21))
# List to store the accuracy for each value of max_depth:
accuracy = []
for depth in max_depth_range:
    
    clf = DecisionTreeClassifier(max_depth = depth, 
                             random_state = 0)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    accuracy.append(score)
print(print(pd.DataFrame(accuracy,max_depth_range,columns =['Accuray Score'])))

#we go for a depth of three, we show the three:
clf = DecisionTreeClassifier(max_depth = 3, random_state = 0)
clf.fit(X_train, y_train)
tree.plot_tree(clf, feature_names=['sepal-length','sepal-width', 'petal-length', 'petal-width'],  class_names=['setosa', 'versicolor', 'class3'], filled=True)
    
# for a depth of 3, we can show the confusion matrix and the classification report
#confusion matrix
y_predict=clf.predict(X_test)
result = metrics.confusion_matrix(y_test, y_predict)
cmd = metrics.ConfusionMatrixDisplay(result, display_labels=['Setosa', 'Versicolor','Virginica'])
print("Confusion Matrix:")
print(result)
cmd.plot()
#classification report
print(metrics.classification_report(y_test, y_predict))

#based on ten draws: 
max_depth_range = list(range(1,10))
scores_list = []
for k in max_depth_range:
    tree = DecisionTreeClassifier(max_depth = k, random_state = 0)
    cv = KFold(n_splits=10, random_state=k, shuffle=True)
    scores = cross_val_score(tree, X, y, scoring='accuracy', cv=cv)
    scores_list.append(np.mean(scores)) 
plt.clf()
plt.plot(max_depth_range, scores_list)
plt.xlabel('Depth of Tree d')
plt.ylabel('Mean Accuracy Score over 10 folds')
plt.title('Accuracy Scores of Tree for depth of d based on 10 folds')
plt.show()
#for each d show the average accuracy score for 10 folds
print(pd.DataFrame(scores_list,max_depth_range,columns =['Accuray Score']))



##SUPPORT VECTOR MACHINE##########################################################################

#using iris sepal features uniquely - two dimensions to start with
from sklearn import svm

#using the four features
svc = svm.SVC(kernel='linear', C=1.0)
svc.fit(X_train, y_train)
score = svc.score(X_test, y_test)
print(score)

# we show the associated confusion matrix and classification report
# confusion matrix
y_predict=svc.predict(X_test)
result = metrics.confusion_matrix(y_test, y_predict)
cmd = metrics.ConfusionMatrixDisplay(result, display_labels=['Setosa', 'Versicolor','Virginica'])
print("Confusion Matrix:")
print(result)
cmd.plot()
# classification report
print(metrics.classification_report(y_test, y_predict))

#accuracy based on 10 folds:
svc = svm.SVC(kernel='linear', C=1.0).fit(X_train, y_train)
cv = KFold(n_splits=10, random_state=0, shuffle=True)
scores = cross_val_score(svc, X, y, scoring='accuracy', cv=cv)
print(scores)
# report performance - mean accuracy score over the 10 folds and its standard deviation
print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))



## LOGISTIC REGRESSION ##########################################################################
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)
model.predict(X_test)
score = model.score(X_test, y_test)
print(score)



## HORSE RACE ##########################################################################
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn import metrics


model_list=["KNN","Tree","SVM","Logistic"]
scores_list = []

knn = KNeighborsClassifier(n_neighbors= 8)
tree = DecisionTreeClassifier(max_depth = 8)
sv = svm.SVC(kernel='linear', C=1.0)
logistic = LogisticRegression(multi_class='multinomial', solver='lbfgs')

cv = KFold(n_splits=10, random_state=0, shuffle=True)

scores = cross_val_score(knn, X, y, scoring='accuracy', cv=cv)
scores_list.append(np.mean(scores))
scores = cross_val_score(tree, X, y, scoring='accuracy', cv=cv)
scores_list.append(np.mean(scores))
scores = cross_val_score(sv, X, y, scoring='accuracy', cv=cv)
scores_list.append(np.mean(scores))
scores = cross_val_score(logistic, X, y, scoring='accuracy', cv=cv)
scores_list.append(np.mean(scores))

#show the average accuracy score for 10 folds
print(pd.DataFrame(scores_list,model_list,columns =['Accuray Score']))



###########################################################################
## EARNINGS RESTATEMENT####################################################
###########################################################################

#relevant imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


## earnings mistatements dataset
dataset=pd.read_csv('restatement_dataset.csv',index_col=False)
# drop lines with missing values
dataset=dataset.dropna()
# get to know the variables
dataset.info()
dataset.describe()

# we first create a dataframe of variables we think may have a predictive power
X = dataset.drop(['any_restatement'], axis=1)
## restrict list of predictors
X= X[['eindex','sum_own','tobins_q', 'ln_at','debt_ratio','capex','rd_to_at_zero','return_on_asset', 'instown_perc']]
print(X.shape)
X.head()

# we create a dataframe of what we want to predict: whether there is misstatement
y = dataset['any_restatement']
print(y.shape)
print(y.value_counts())

# #we form the train and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
print("Size of the train sample of features is :", X_train.shape)
print("Size of the train sample of species (labels) is :", y_train.shape)
print("Size of the test sample of features is :", X_test.shape)
print("Size of the test sample of species (labels) is :", y_test.shape)

# let us try out one classifier first - the KNN one
# we try out a KNN classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_predict=knn.predict(X_test)
result = metrics.confusion_matrix(y_test, y_predict)

#confusion matrix
cmd = metrics.ConfusionMatrixDisplay(result)
print("Confusion Matrix:")
print(result)
cmd.plot()

#classification report
print(metrics.classification_report(y_test, y_predict))

#to isolate the percentage of positive cases predicted to the total number of positive cases, we go:
print(metrics.recall_score(y_test, y_predict, pos_label=1, average="binary"))

#alternative
print(metrics.precision_recall_fscore_support(y_test, y_predict)[1][1])

#Best KNN model
from sklearn.metrics import recall_score, make_scorer
ftwo_scorer = make_scorer(recall_score, pos_label=1, average="binary")

k_range = list(range(1,20))
scores_list = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv = KFold(n_splits=10, random_state=k, shuffle=True)
    scores = cross_val_score(knn, X, y, scoring=ftwo_scorer, cv=cv)
    scores_list.append(np.mean(scores))
plt.clf()
plt.plot(k_range, scores_list)
plt.xlabel('Value of k for KNN')
plt.ylabel('Recall score over 10 folds')
plt.title('Recall scores of KNN for values of k based on 10 folds')
plt.show()

#for each k show the average recall score for 10 folds
print(pd.DataFrame(scores_list,k_range,columns =['Recall Score']))

# same exercise for decision trees
ftwo_scorer = make_scorer(recall_score, pos_label=1, average="binary")

max_depth_range = list(range(1,21))
scores_list = []
for k in max_depth_range:
    tree = DecisionTreeClassifier(max_depth = k, random_state = 0)
    cv = KFold(n_splits=10, random_state=k, shuffle=True)
    scores = cross_val_score(tree, X, y, scoring=ftwo_scorer, cv=cv)
    scores_list.append(np.mean(scores))
plt.clf()
plt.plot(max_depth_range, scores_list)
plt.xlabel('Depth of Tree d')
plt.ylabel('Mean recall Score over 10 folds')
plt.title('Recall scores of Tree for depth of d based on 10 folds')
plt.show()

#for each d show the average recall score for 10 folds
print(pd.DataFrame(scores_list,max_depth_range,columns =['Recall Score']))

#importance of the features
tree = DecisionTreeClassifier(max_depth = 19, random_state = 0)
tree.fit(X_train,y_train)
importances = tree.feature_importances_
names = list(X_train.columns.values)
imp = {'Feature Name': names,'Importance': importances}
imp = pd.DataFrame(imp)
print(imp)

#we now run a horse race
from sklearn.linear_model import LogisticRegression
from sklearn import svm  

ftwo_scorer = make_scorer(recall_score, pos_label=1, average="binary")

model_list=["KNN","Tree","SVM","Logistic"]
scores_list = []

knn = KNeighborsClassifier(n_neighbors=1)
tree = DecisionTreeClassifier(max_depth = 19)
sv = svm.SVC(kernel='linear', C=1.0)
logistic = LogisticRegression()

cv = KFold(n_splits=10, random_state=0, shuffle=True)

scores = cross_val_score(knn, X, y, scoring=ftwo_scorer, cv=cv)
scores_list.append(np.mean(scores))
scores = cross_val_score(tree, X, y, scoring=ftwo_scorer, cv=cv)
scores_list.append(np.mean(scores))
scores = cross_val_score(sv, X, y, scoring=ftwo_scorer, cv=cv)
scores_list.append(np.mean(scores))
scores = cross_val_score(logistic, X, y, scoring=ftwo_scorer, cv=cv)
scores_list.append(np.mean(scores))

#show the average accuracy score for 10 folds
print(pd.DataFrame(scores_list,model_list,columns =['Recall Score']))
