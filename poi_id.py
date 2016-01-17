#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary','deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

'''
### Task 2: Remove outliers
### Removing outliers depends on the nature of the variable - plotting them (via histogram) so see distribution of
### the data and spot (more easily) outliers

import matplotlib.pyplot
import numpy

# needed to remake a list of features because it didn't like plotting the 'poi' variable
outliers_test = ['salary','deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi']
for x in outliers_test:
    test = [x]
    data = featureFormat(data_dict, test)
    data1 = numpy.ravel(data)
    # print the length of the data (to see how many datapoints there are for that variable) to screen
    print len(data1)
    # the histogram of the data
    # turn of plots when running through the script
    #matplotlib.pyplot.hist(data1)
    #matplotlib.pyplot.xlabel(x)
    #matplotlib.pyplot.show()

## negative values - deferral_payments, restricted_stock_deferred, deferred_income, total_stock_value, restricted_stock


### not all have a good amount of non-missing data:
    ### loan_advances only has 4 data points
    ### directors_fees only have 8 data points
    ### might consider removing these are this is so few data

### that works to plot all the data - for most of the financial there is an obvious outlier
for value in outliers_test:
    max_value = max(data_dict, key=lambda v: data_dict[v][value] if isinstance(data_dict[v][value],int) else float("-inf"))
    print(max_value, value)

for value in outliers_test:
    min_value = min(data_dict, key=lambda v: data_dict[v][value] if isinstance(data_dict[v][value],int) else float("-inf"))
    print(min_value, value)

'''

### Ah! This is TOTAL - need to remove that
data_dict.pop("TOTAL", None)
## also remove the travel agency that is in the dataset
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", None)

'''
### Let's look at this again and see what the new distributions are
for x in outliers_test:
    test = [x]
    data = featureFormat(data_dict, test)
    data1 = numpy.ravel(data)
    # print the length of the data (to see how many datapoints there are for that variable) to screen
    print len(data1)
    # the histogram of the data
    # turn of plots when running through the script
    #matplotlib.pyplot.hist(data1)
    #matplotlib.pyplot.xlabel(x)
    #matplotlib.pyplot.show()

### not all have a good amount of non-missing data with TOTAL now removed:
    ### loan_advances only has 3 data points
    ### directors_fees only have 7 data points
    ### might consider removing these are this is so few data

### let's now see who is the max for each variable of interest
for value in outliers_test:
    max_value = max(data_dict, key=lambda v: data_dict[v][value] if isinstance(data_dict[v][value],int) else float("-inf"))
    print(max_value, value)

# get an idea of missing per variable of interest:
print len(data_dict)
for value in outliers_test:
    count_NaN_tp = 0
    count_NaN_non = 0
    count_poi = 0
    for key in data_dict.keys():
        if data_dict[key]['poi'] == True:
            count_poi+=1
        if data_dict[key][value] == 'NaN' and data_dict[key]['poi'] == True :
            count_NaN_tp+=1
        if data_dict[key][value] == 'NaN' and data_dict[key]['poi'] == False :
            count_NaN_non+=1
    print value
    print count_poi
    print count_NaN_tp
    print count_NaN_non
    print float(count_NaN_tp)/float(count_poi)
    print float(count_NaN_tp)/len(data_dict.keys())
    print float(count_NaN_non)/len(data_dict.keys())

# 145 people in total with 18 POIs -
# most are missing deferral_payments (13/18), loan_advances (17/18), restricted_stock_deferred (18/18), director_fees (18/18),
# Over 50% of POIs are missing these variables of interest - so these might not be informative


'''


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# explore the dataset
# total number of datapoint
print len(my_dataset)

# allocation across classes (POI/non-POI)
count_poi = 0
count_not_poi = 0
for key in my_dataset.keys():
    if my_dataset[key]['poi'] == True:
        count_poi += 1
    else:
        count_not_poi += 1
print count_poi
print count_not_poi
print float(count_poi) / len(my_dataset)

# missingness per datapoint
count_missing_dp = dict()
count = 0
for key in my_dataset.keys():
    for stuff in features_list:
        if my_dataset[key][stuff] == 'NaN':
            count += 1
    #print key, count
    try:
        count_missing_dp[key].append(count)
    except KeyError:
        count_missing_dp[key] = count
    count = 0

## missing per feature:
count_missing_f = dict()
count = 0
for key in my_dataset.keys():
    for stuff in features_list:
        if my_dataset[key][stuff] == 'NaN':
            count += 1
        try:
            count_missing_f[stuff].append(count)
        except KeyError:
            count_missing_f[stuff] = [count]
        count = 0

missing_features =  {k:sum(v) for k,v in count_missing_f.items()}


# create two new features about the proportion of emails from and to a person and a poi
# with the logic that poi might have different propotional rate of interaction than other individuals.
# NaN values are being problematic! - need to set all values to 0

for key in my_dataset.keys():
    my_dataset[key]['from_this_person_to_poi_ratio'] = 0
    my_dataset[key]['to_this_person_from_poi_ratio'] = 0
    for stuff in features_list:
        if my_dataset[key]['from_messages'] == 'NaN':
            my_dataset[key][stuff] = 0
        if my_dataset[key]['to_messages'] == 'NaN':
            my_dataset[key][stuff] = 0
    # divide by zero issue
    if my_dataset[key]['from_messages'] > 0:
        my_dataset[key]['from_this_person_to_poi_ratio'] = ( float(my_dataset[key]['from_this_person_to_poi']) / float(my_dataset[key]['from_messages']) )
    if my_dataset[key]['to_messages'] > 0:
        my_dataset[key]['to_this_person_from_poi_ratio'] = ( float(my_dataset[key]['from_poi_to_this_person']) / float(my_dataset[key]['to_messages']) )


# put the new features into the feature list
features_list = ['poi', 'salary','deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi', 'to_this_person_from_poi_ratio', 'from_this_person_to_poi_ratio']


### Extract features and labels from dataset for local testing

from sklearn.cross_validation import train_test_split
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline


feature_train, feature_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.1, random_state=42)

folds = 10
cv = cross_validation.StratifiedShuffleSplit(labels, folds, random_state = 42)


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split=2, criterion="entropy")
clf = clf.fit(feature_train, labels_train)

clf = DecisionTreeClassifier()
pipeline = Pipeline([('kbest',SelectKBest(f_regression)), ('DTC', clf)])

param_grid = dict(kbest__k=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                  DTC__min_samples_split=[2,3,4,5,6],
                  DTC__criterion=("entropy", "gini")
                  )

grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv,  scoring = 'recall')
grid_search.fit(features, labels)
print(grid_search.best_estimator_)

clf=grid_search.best_estimator_

print(grid_search.best_params_, grid_search.best_score_)

dump_classifier_and_data(clf, my_dataset, features_list)

best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    if param_name == 'kbest__k':
        number_features =  best_parameters[param_name]

####
selector = SelectKBest(f_regression, k=number_features)
selectedFeatures = selector.fit(features,labels)

# get the names of the selected features
feature_names = [features_list[i] for i in selectedFeatures.get_support(indices=True)]
feature_list = feature_names
print feature_list

## this is from here: http://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/
print "Features sorted by their score:"
print sorted(zip(map(lambda x: round(x, 4), selectedFeatures.scores_),
                 feature_list), reverse=True)

# missing per feature used
## missing per feature:
count_missing_f_again = dict()
count = 0
for key in my_dataset.keys():
    for stuff in feature_list:
        if my_dataset[key][stuff] == 'NaN':
            count += 1
        try:
            count_missing_f_again[stuff].append(count)
        except KeyError:
            count_missing_f_again[stuff] = [count]
        count = 0

missing_features_again =  {k:sum(v) for k,v in count_missing_f_again.items()}
print missing_features_again

'''
## re-run model without features that I created
## create two features and one was selected in the final step: to_this_person_from_poi_ratio

# feature list without the features I created
features_list = ['poi', 'salary','deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi']


### Extract features and labels from dataset for local testing

from sklearn.cross_validation import train_test_split
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline


feature_train, feature_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.1, random_state=42)

folds = 10
cv = cross_validation.StratifiedShuffleSplit(labels, folds, random_state = 42)


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split=2, criterion="entropy")
clf = clf.fit(feature_train, labels_train)

clf = DecisionTreeClassifier()
pipeline = Pipeline([('kbest',SelectKBest(f_regression)), ('DTC', clf)])

param_grid = dict(kbest__k=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
                  DTC__min_samples_split=[2,3,4,5,6],
                  DTC__criterion=("entropy", "gini")
                  )

grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv,  scoring = 'recall')
grid_search.fit(features, labels)
print(grid_search.best_estimator_)

clf=grid_search.best_estimator_

print(grid_search.best_params_, grid_search.best_score_)

dump_classifier_and_data(clf, my_dataset, features_list)

best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    if param_name == 'kbest__k':
        number_features =  best_parameters[param_name]

####
selector = SelectKBest(f_regression, k=number_features)
selectedFeatures = selector.fit(features,labels)

# get the names of the selected features
feature_names = [features_list[i] for i in selectedFeatures.get_support(indices=True)]
feature_list = feature_names
print feature_list

## this is from here: http://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/
print "Features sorted by their score:"
print sorted(zip(map(lambda x: round(x, 4), selectedFeatures.scores_),
                 feature_list), reverse=True)

'''

'''
## LogisticRegression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
pipeline = Pipeline([('kbest',SelectKBest(f_regression)),('logL', clf)])

param_grid = dict(kbest__k=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                  logL__C=[0.01, 0.1,1,10,100,1000],
                  logL__tol=[10**-128,10**-64, 10**-32,10**-16,10**-8,10**-4,10**-2]
                  )


### ADABOOST
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()
pipeline = Pipeline([('kbest',SelectKBest(f_regression)), ('Ada', clf)])

param_grid = dict(kbest__k=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                  Ada__n_estimators=[40,50,60,70,80,90]
                  )



from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
clf = LinearSVC()
pipeline = Pipeline([ ('minmaxer', MinMaxScaler()), ('kbest',SelectKBest(f_regression)), ('svc', clf)])

param_grid = dict(
                    kbest__k=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                  svc__C=[0.01, 0.1,1,10,100,1000],
                  svc__tol=[10**-256, 10**-128,10**-64, 10**-32,10**-16,10**-8,10**-4,10**-2]
                  )

grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=cv,  scoring = 'recall')
grid_search.fit(features, labels)
print(grid_search.best_estimator_)

clf=grid_search.best_estimator_

print(grid_search.best_params_, grid_search.best_score_)

dump_classifier_and_data(clf, my_dataset, features_list)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html



# Provided to give you a starting point. Try a variety of classifiers.


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!




### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
'''