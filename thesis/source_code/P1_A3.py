

### Import of used libraries ###


import os
import random as rd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
scaler = StandardScaler()

import tensorflow as tf
from tensorflow import keras


data_path = '../raw_data'
cwd = os.path.dirname(__file__)
raw_data_folder = os.path.abspath(os.path.join(cwd, data_path))


def load_df(name):
    return pd.read_csv(os.path.join(raw_data_folder, name))



# Laden der Daten
df_working = load_df('adapted_data.csv').drop(['Unnamed: 0'], axis =1)

df_test = df_working.copy(deep = True)

# Merkmale fr erstes Jahr 
features_year1 = ['styria_dummy', 'not_styria_dummy', 'germany_dummy',
                  'num_parallel_studies', 'years_since_matura', 'firstGen',
                  'geschlecht', 'AHS_dummy', 'BHS_dummy', 'ausland_vorbildung_dummy',
                  'sonstige_vorbildung_dummy', 'jus_dummy', 'bwl_dummy',
                  'delayed_dummy','active_3years', 'ECTS_year', 'active_dummy']


# Merkamle fr Jahr >= 2
features_years = ['Studienjahr', 'styria_dummy', 'not_styria_dummy', 'germany_dummy',
                  'num_parallel_studies', 'cum_ects_pos_before', 'avgECTS_sem_before', 'ects_year_before',
                  'full_duration_sem_before', 'geschlecht', 'years_since_matura', 'firstGen', 'AHS_dummy',
                  'BHS_dummy', 'ausland_vorbildung_dummy', 'sonstige_vorbildung_dummy', 'delayed_dummy',
                  'jus_dummy', 'bwl_dummy', 'active_3years', 'ECTS_year', 'active_dummy']



#############################################################################################################################



### Helper Function, for getting labels or 3 Years in the future ###

def active_3years(df):
    count = 0
  
    for i in range(len(df)):
        
        matrikel_number = df.loc[i, 'matrikel_num']
        studienjahr = df.loc[i, 'Studienjahr'] + 2
        jus_dummy = df.loc[i, 'jus_dummy']
        bwl_dummy = df.loc[i, 'bwl_dummy']
        
        student_in_3_years = df.query('matrikel_num == @matrikel_number and Studienjahr == @studienjahr and jus_dummy == @jus_dummy and bwl_dummy == @bwl_dummy').reset_index(drop = True)
        
        if len(student_in_3_years) != 0:
            count += 1
            active = student_in_3_years.loc[0, 'active_dummy']
            df.loc[i, 'active_3years'] = active
        else:
            df.loc[i, 'active_3years'] = 0 # mache ich auch null, weil er ja nicht aktiv ist, wenn er nicht mehr da ist.
        
        
    # print(count)  
    return df


df_new = active_3years(df_test).query('year <= 17')




#######################################################################################################################################


### Helper Functions fr zusammenstellen der Daten ###

# Jahr 1
def create_data1(df):
    df = df.query('Studienjahr == 1')[features_year1].copy(deep = True)
    df = df.dropna().reset_index(drop = True)
    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.1, random_state = 42)
    for train_index, test_index in split.split(df, df['active_3years']):
        df_train = df.loc[train_index, :]
        df_test = df.loc[test_index, :]
    df_train_copy = df_train.copy(deep = True)
    y_train = list(df_train['active_3years'])
    y_test = list(df_test['active_3years'])
    df_train = scaler.fit_transform(df_train.drop(['active_3years', 'ECTS_year', 'active_dummy'], axis = 1))
    df_test = scaler.fit_transform(df_test.drop(['active_3years', 'ECTS_year', 'active_dummy'], axis = 1))
    return df_train, y_train, df_test, y_test, df_train_copy


# Jahre >= 2
def create_data2(df):
    df = df.query('Studienjahr > 1')[features_years].copy(deep = True)
    df = df.dropna().reset_index(drop = True)
    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.1, random_state = 42)
    for train_index, test_index in split.split(df, df['active_3years']):
        
        df_train = df.loc[train_index, :]
        df_test = df.loc[test_index, :]
    df_train_copy = df_train.copy(deep = True)
    y_train = list(df_train['active_3years'])
    y_test = list(df_test['active_3years'])
    df_train = scaler.fit_transform(df_train.drop(['active_3years', 'ECTS_year', 'active_dummy'], axis = 1))
    df_test = scaler.fit_transform(df_test.drop(['active_3years', 'ECTS_year', 'active_dummy'], axis = 1))
    return df_train, y_train, df_test, y_test, df_train_copy


# Tatschliches Erstellen
df1_train, y1_train, df1_test, y1_test, df1_train_copy = create_data1(df_new)
df2_train, y2_train, df2_test, y2_test, df2_train_copy = create_data2(df_new)

# print(len(df1_train))
# print(len(df2_train))

##############################################################################################################################

### Helper Functions fr das Trainieren und Auswerten ###

def display_scores(scores):
    print('Scores: ', scores)
    print('Mean: ', scores.mean())
    print('Standard deviation: ', scores.std())


def perform_classification(classifier, df, labels):
    df = df.copy()
    print('#####')
    scores = cross_val_score(classifier, df, labels, cv=5, scoring='accuracy')
    display_scores(scores)
    
    predicted = cross_val_predict(classifier, df, labels, cv=5)
    matrix = confusion_matrix(labels, predicted)
    print(matrix)

    plt.show()  
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_matrix = matrix / row_sums
    np.fill_diagonal(norm_matrix, 0)

    plt.show()    
    print('#####')
    return


def probability_classification(classifier, df_train, y_train, df_test, y_test):
    classifier = sklearn.base.clone(classifier)
    classifier.fit(df_train, y_train)
    probabilities = classifier.predict_proba(df_test)[:,1]
    prediction = np.sum(probabilities)
    real_value = np.sum(y_test)
    print('Prediction: ' + str(prediction) + ' vs. Real value: ' + str(real_value))



########################################################################################################################################

### Logistic Regression ### 
from sklearn.linear_model import LogisticRegression


print('Logistic Regression')
log_clf1 = LogisticRegression(random_state=0, multi_class = 'multinomial')
log_clf2 = LogisticRegression(random_state=0, multi_class = 'multinomial')


probability_classification(log_clf1, df1_train, y1_train, df1_test, y1_test)
probability_classification(log_clf2, df2_train, y2_train, df2_test, y2_test)

perform_classification(log_clf1, df1_train, y1_train)
perform_classification(log_clf2, df2_train, y2_train)
print('')


### Support Vector Machines ###

from sklearn.svm import SVC

svm_clf1 = SVC(kernel = 'rbf', gamma = 10, C = 100, probability = True)
svm_clf2 = SVC(kernel = 'rbf', gamma = 10, C = 100, probability = True)
print('')
print('SVM')
probability_classification(svm_clf1, df1_train, y1_train, df1_test, y1_test)
probability_classification(svm_clf2, df2_train, y2_train, df2_test, y2_test)

perform_classification(svm_clf1, df1_train, y1_train)
perform_classification(svm_clf2, df2_train, y2_train)
print('')



### Random Forest ###
from sklearn.ensemble import RandomForestClassifier

forest_clf1 = RandomForestClassifier()
forest_clf2 = RandomForestClassifier()

print('')
print('Random Forest')
probability_classification(forest_clf1, df1_train, y1_train, df1_test, y1_test)
probability_classification(forest_clf2, df2_train, y2_train, df2_test, y2_test)

perform_classification(forest_clf1, df1_train, y1_train)
perform_classification(forest_clf2, df2_train, y2_train)
print('')


### KNN ###

print('KNN')
model1 = keras.models.Sequential([
        keras.layers.Dense(60, activation = 'relu', input_shape = df1_train.shape[1:]),
        keras.layers.Dense(40, activation = 'relu'),
        keras.layers.Dense(20, activation = 'relu'),
        keras.layers.Dense(1, activation = 'sigmoid')
    ])

model1.compile(loss = 'binary_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])
history1 = model1.fit(df1_train, np.array(y1_train), epochs = 35, validation_split = 0.1)

predictions1 = model1.predict(df1_test)
predict1 = np.sum(predictions1)
real1 = np.sum(y1_test)
print('Prediction: ' + str(predict1) + ' vs. Real value: ' + str(real1))


model2 = keras.models.Sequential([
        keras.layers.Dense(60, activation = 'relu', input_shape = df2_train.shape[1:]),
        keras.layers.Dense(40, activation = 'relu'),
        keras.layers.Dense(20, activation = 'relu'),
        keras.layers.Dense(1, activation = 'sigmoid')
    ])
model2.compile(loss = 'binary_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])
history2 = model2.fit(df2_train, np.array(y2_train), epochs = 35, validation_split = 0.1)

predictions2 = model2.predict(df2_test)
predict2 = np.sum(predictions2)
real2 = np.sum(y2_test)
print('Prediction: ' + str(predict2) + ' vs. Real value: ' + str(real2))  
