

### importing libraries ###


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

### loading data ###
data_path = '../raw_data'
cwd = os.path.dirname(__file__)
raw_data_folder = os.path.abspath(os.path.join(cwd, data_path))


def load_df(name):
    return pd.read_csv(os.path.join(raw_data_folder, name))


features_year1 = [
    'styria_dummy', 'not_styria_dummy',
    'germany_dummy', 'num_parallel_studies',
    'years_since_matura', 'firstGen',
    'geschlecht', 'AHS_dummy',
    'BHS_dummy', 'ausland_vorbildung_dummy',
    'sonstige_vorbildung_dummy', 'jus_dummy',
    'bwl_dummy', 'delayed_dummy',
    'active_3years', 'ECTS_year',
    'active_dummy'
    ]

features_years = [
    'Studienjahr', 'styria_dummy',
    'not_styria_dummy', 'germany_dummy',
    'num_parallel_studies', 'cum_ects_pos_before',
    'avgECTS_sem_before', 'ects_year_before',
    'full_duration_sem_before', 'geschlecht',
    'years_since_matura', 'firstGen',
    'AHS_dummy', 'BHS_dummy',
    'ausland_vorbildung_dummy', 'sonstige_vorbildung_dummy',
    'delayed_dummy', 'jus_dummy',
    'bwl_dummy', 'active_3years',
    'ECTS_year', 'active_dummy'
    ]


df = load_df('adapted_data.csv').drop(['Unnamed: 0'], axis =1)



### helper Function for creating new labels ###

def active_2years(df):
    df['active_3years'] = 0
    for i in range(len(df)):
        
        matrikel_number = df.loc[i, 'matrikel_num']
        studienjahr = df.loc[i, 'Studienjahr'] + 1
        jus_dummy = df.loc[i, 'jus_dummy']
        bwl_dummy = df.loc[i, 'bwl_dummy']
        
        student2 = df.query('matrikel_num == @matrikel_number and Studienjahr == @studienjahr and jus_dummy == @jus_dummy and bwl_dummy == @bwl_dummy').reset_index(drop = True)

        if len(student2) != 0:
            active = student2.loc[0, 'active_dummy']
            df.loc[i, 'active_2years'] = active
            
    return df

### making new DataFrames ###

df_2years = active_2years(df)
df_2years['active_2years'].fillna(0)


df = load_df('adapted_data.csv').drop(['Unnamed: 0'], axis =1)
df_1year = df.copy(deep = True)
df_1year['active_dummy'].fillna(0)



### New Training Data ### 

df2years = df_2years[features2years].query('Studienjahr == 1').reset_index(drop = True).dropna()
df1year = df_1year[features1year].query('Studienjahr == 1').reset_index(drop = True).dropna()


df_train2 = df2years.drop(['active_2years'], axis = 1)
y_train2 = df2years['active_2years']

df_train1 = df1year.drop(['active_dummy'], axis = 1)
y_train1 = df1year['active_dummy']


### choosing SVM Classifier ### 
from sklearn.svm import SVC
svm_clf1year = SVC(kernel = 'rbf', gamma = 10, C = 100, probability = True)
svm_clf2years = SVC(kernel = 'rbf', gamma = 10, C = 100, probability = True)

svm_clf1year.fit(df_train1, y_train1)
svm_clf2years.fit(df_train2, y_train2)


### making real and dummy DataFrames ###

df1_15 = df_1year.query('Studienjahr == 1 and year == 15')[features1year].drop(['active_dummy'], axis = 1).fillna(0)
df1_16 = df_1year.query('Studienjahr == 1 and year == 16')[features1year].drop(['active_dummy'], axis = 1).fillna(0)
df1_17 = df_1year.query('Studienjahr == 1 and year == 17')[features1year].drop(['active_dummy'], axis = 1).fillna(0)

df1_dummy16 = df1_15.sample(n = 2048)
df1_dummy17 = df1_16.sample(n = 1899)


### helper Function for making Prediction ###

def predict_sum(classifier, df):
    probabilities = classifier.predict_proba(df)[:,1]
    for i in range(len(probabilities)):
        num = rd.random()
        if probabilities[i] >= num:
            probabilities[i] = 1
        else:
            probabilities[i] = 0
    print(sum(probabilities))
    print('######')



### Making actually Predictions ###

print('Schtzung echte Daten: ')
predict_sum(svm_clf1year, df1_16)
print('Schtzung dummy Daten: ')
predict_sum(svm_clf1year, df1_dummy16)

print('Tatschliche Anzahl: ')
print(len(df.query('Studienjahr == 1 and year == 16 and active_dummy == 1')))
print('')

print('Schtzung echte Daten: ')
predict_sum(svm_clf1year, df1_17)
print('Schtzung dummy Daten: ')
predict_sum(svm_clf1year, df1_dummy17)

print('Tatschliche Anzahl: ')
print(len(df.query('Studienjahr == 1 and year == 17 and active_dummy == 1')))
print('')




### data for estimation over 2 years ###

df2_15 = df_2years.query('Studienjahr == 1 and year == 15 and active_2years >= 0')[features2years].drop(['active_2years'], axis = 1).fillna(0)
df2_16 = df_2years.query('Studienjahr == 1 and year == 16 and active_2years >= 0')[features2years].drop(['active_2years'], axis = 1).fillna(0)
df2_17 = df_2years.query('Studienjahr == 1 and year == 17 and active_2years >= 0')[features2years].drop(['active_2years'], axis = 1).fillna(0)

df2_dummy16 = df2_15.sample(n = 1421)
df2_dummy17 = df2_16.sample(n = 1249)


### Making actually Predicitions ###

print('Schtzung echte Daten: ')
predict_sum(svm_clf2years, df2_16)
print('Schtzung dummy Daten: ')
predict_sum(svm_clf2years, df2_dummy16)

print('Tatschliche Anzahl: ')
print(len(df.query('Studienjahr == 2 and year == 17 and active_dummy == 1')))
print('')

print('Schtzung echte Daten: ')
predict_sum(svm_clf2years, df2_17)
print('Schtzung dummy Daten: ')
predict_sum(svm_clf2years, df2_dummy17)

print('Tatschliche Anzahl: ')
print(len(df.query('Studienjahr == 2 and year == 18 and active_dummy == 1')))
print('')
