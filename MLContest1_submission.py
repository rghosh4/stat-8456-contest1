# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:05:01 2023

@author: rghosh
"""

import pandas as pd 
import numpy as np
import os
from sklearn.model_selection import train_test_split

## Updating the Directory and just running the Entire Script should work 
os.chdir("C:/Users/rghosh/Documents/Graduate Curicullum/Spring'23/STAT 8456/Contest 1/unodatamining-2023-1")

# Import the data
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
# train.head(2)
# print(train.dtypes)
# print(train.describe())
# print(train.info())


###########################
# Convert the orderDate and deliveryDate columns to datetime objects
train['orderDate'] = pd.to_datetime(train['orderDate'])
train['deliveryDate'] = pd.to_datetime(train['deliveryDate'])
train['dateOfBirth'] = pd.to_datetime(train['dateOfBirth'],errors='coerce')
train['creationDate'] = pd.to_datetime(train['creationDate'],errors='coerce')

# Calculate the time to delivery in days
train['time_to_delivery'] = (train['deliveryDate'] - train['orderDate']).dt.days
train['time_to_delivery'].max()
print(train['deliveryDate'].min())
print(train['deliveryDate'].isnull().sum())
## numeric imputing trying to preserve their different characteristic
train.loc[train['deliveryDate'] == pd.Timestamp('1990-12-31'), 'time_to_delivery'] = 9999
train.loc[train['deliveryDate'].isna(), 'time_to_delivery'] = 99999
##############################


################## Checking Levels of Prospective Categorical Columns #####################

##item id ##

train['itemID'].unique()
train['itemID'].value_counts()



##size##
train['size'].unique()
train['size'].value_counts()


##color##
train['color'].unique()
train['color'].value_counts()

##manufacturer id ##

train['manufacturerID'].unique()
train['manufacturerID'].value_counts()


## customer id ##
train['customerID'].unique()


## salutation ##

train['salutation'].unique()
train['salutation'].value_counts()

#############################################

## price ##
print(train['price'].describe())


## date of birth to age ##
## compute age at order ##
train['age_at_order'] = (train['orderDate'] - train['dateOfBirth']) / np.timedelta64(1, 'Y')
train.groupby('salutation').size()
train.groupby('salutation')['dateOfBirth'].apply(lambda x: x.isnull().sum())
train['age_at_order'] = train.groupby('salutation')['age_at_order'].transform(lambda x: x.fillna(x.median()))

#min_age_idx = train['age_at_order'].idxmin()
#train.loc[min_age_idx]
#0.08  for a female .. bad data 


## creation date to age_in_system
## numeric imputing trying to preserve their different characteristic 
train['age_in_system'] = (train['creationDate'] - train['orderDate']).dt.days
train.loc[train['age_in_system'].isna(), 'age_in_system'] = 99999


####### Final Columns ##########

# Create a list of object columns
object_cols = ['itemID', 'size', 'color','manufacturerID','customerID','salutation','state']
#object_cols = ['itemID', 'size', 'color','manufacturerID','customerID','salutation','state','return']
#'return' to be in numeric 

# Use apply() to convert object columns to categorical data type
train[object_cols] = train[object_cols].apply(lambda x: x.astype('category'))

## more work needed on cleaning color, size and salutation string to float error 

X = train[['itemID','manufacturerID','price','customerID',
          'time_to_delivery','age_at_order','age_in_system']]
y = train['return']

numerical_features = X.select_dtypes(include='number').columns.tolist()
categorical_features = X.select_dtypes(exclude='number').columns.tolist()

# Train test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)




######### Pipe line ###############
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('scale', MaxAbsScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('one-hot', OneHotEncoder(drop='first',handle_unknown='ignore', sparse=True))
])

full_processor = ColumnTransformer(transformers=[
    ('number', numeric_pipeline, numerical_features),
    ('category', categorical_pipeline, categorical_features)
])

full_processor.fit_transform(X_train)


# Import necessary modules
from scipy.stats import randint
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3,5,7,None],
              "max_features": randint(1, 8),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree,param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X_train,y_train)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))

preds_dtc = tree_cv.predict(X_test)
accuracy_score(y_test, preds_dtc)


############################# Final Model on Entire Train Data ##########################
##Fitting the Classifier on whole train with the best parameters obetained  
full_processor.fit_transform(X)
full_data_tree = DecisionTreeClassifier(criterion='entropy',max_depth=7,max_features=6)
full_data_tree.fit(X,y)

##Checking Accuracy on test for sanity check
preds_dtc=full_data_tree.predict(X_test)
accuracy_score(y_test, preds_dtc)


#################### Prediction on Test Set #############################

## Preprocessing the Test Set First with the exact similar steps

# Convert the orderDate and deliveryDate columns to datetime objects
test['orderDate'] = pd.to_datetime(test['orderDate'])
test['deliveryDate'] = pd.to_datetime(test['deliveryDate'])
test['dateOfBirth'] = pd.to_datetime(test['dateOfBirth'],errors='coerce')
test['creationDate'] = pd.to_datetime(test['creationDate'],errors='coerce')

# Calculate the time to delivery in days
test['time_to_delivery'] = (test['deliveryDate'] - test['orderDate']).dt.days
test['time_to_delivery'].max()
print(test['deliveryDate'].min())
print(test['deliveryDate'].isnull().sum())
test.loc[test['deliveryDate'] == pd.Timestamp('1990-12-31'), 'time_to_delivery'] = 9999
test.loc[test['deliveryDate'].isna(), 'time_to_delivery'] = 99999

test['age_at_order'] = (test['orderDate'] - test['dateOfBirth']) / np.timedelta64(1, 'Y')
test.groupby('salutation').size()
test.groupby('salutation')['dateOfBirth'].apply(lambda x: x.isnull().sum())
test['age_at_order'] = test.groupby('salutation')['age_at_order'].transform(lambda x: x.fillna(x.median()))


test['age_in_system'] = (test['creationDate'] - test['orderDate']).dt.days
test.loc[test['age_in_system'].isna(), 'age_in_system'] = 99999

####### Final Columns ##########

# Create a list of object columns
object_cols = ['itemID', 'size', 'color','manufacturerID','customerID','salutation','state']
#'return' to be in numeric 

# Use apply() to convert object columns to categorical data type
test[object_cols] = test[object_cols].apply(lambda x: x.astype('category'))
X_final_test = test[['itemID','manufacturerID','price','customerID',
          'time_to_delivery','age_at_order','age_in_system']]

full_processor.fit_transform(X_final_test)
#################################

## Final Prediction and Submission creation
preds_final = full_data_tree.predict_proba(X_final_test)

#tweak threshold to match event rate of train set
#create submission
submission = pd.DataFrame()
submission['id'] = test['id']
submission['return'] = np.where(preds_final[:,1] > 0.4833, 1, 0)
submission.to_csv('submission1.csv', index=False)

##################### Feature Importance Plot ########################

import matplotlib.pyplot as plt
import seaborn as sns

# Get the feature importances
importances = full_data_tree.feature_importances_
features = X.columns

# Create a dataframe of feature importances
feature_importances = pd.DataFrame({'feature': features, 'importance': importances})
feature_importances = feature_importances.sort_values('importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title('Variable Importance Plot')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

### ROC- AUC ####
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Get predicted probabilities for the positive class (class 1)
y_prob = full_data_tree.predict_proba(X)[:, 1]

# Compute the false positive rate (fpr), true positive rate (tpr), and thresholds for different probability cutoffs
fpr, tpr, thresholds = roc_curve(y, y_prob)

# Calculate the AUC score
auc = roc_auc_score(y, y_prob)

# Plot the ROC curve
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend()
plt.show()



