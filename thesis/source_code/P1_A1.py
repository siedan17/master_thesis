### Import of the used libraries ###

import sys
import os
import random as rd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
from tensorflow import keras

scaler = StandardScaler()

# Definieren des path:
data_path = '../raw_data'
cwd = os.path.dirname(__file__)
raw_data_folder = os.path.abspath(os.path.join(cwd, data_path))

def load_df(name):
    return pd.read_csv(os.path.join(raw_data_folder, name))


# Laden der Daten:
df = load_df('adapted_data.csv').drop(['Unnamed: 0'], axis =1)


# Verwendete Merkmale im ersten Studienjahr: 
features_year1 = ['styria_dummy', 'not_styria_dummy', 'germany_dummy',
                  'num_parallel_studies', 'cum_ects_pos_before', 'years_since_matura', 'firstGen',
                  'geschlecht', 'AHS_dummy', 'BHS_dummy', 'ausland_vorbildung_dummy',
                  'sonstige_vorbildung_dummy', 'jus_dummy', 'bwl_dummy',
                  'delayed_dummy', 'ECTS_year', 'active_dummy']

# Verwendete Merkmale for Studienjahre:
features_years = ['Studienjahr', 'styria_dummy', 'not_styria_dummy', 'germany_dummy',
                  'num_parallel_studies', 'cum_ects_pos_before', 'avgECTS_sem_before', 'ects_year_before',
                  'full_duration_sem_before', 'geschlecht', 'years_since_matura', 'firstGen', 'AHS_dummy',
                  'BHS_dummy', 'ausland_vorbildung_dummy', 'sonstige_vorbildung_dummy',
                  'delayed_dummy', 'jus_dummy', 'bwl_dummy',
                  'ECTS_year', 'active_dummy']

###############################################################################################################################

### Help Functions and Training Functions ###

def active(df_, prediction):
    df = df_.copy(deep = True)
    df.reset_index(drop = True)
    if len(df) == len(prediction):
        new_column = []
        for i in range(len(prediction)):
            if prediction[i] >= 16:
                new_column.append(1)
            else:
                new_column.append(0)
        df.insert(len(df.columns), 'active_predicted', new_column, allow_duplicates = True)
        df.insert(len(df.columns), 'ECTS_predicted', prediction, allow_duplicates = True)
    else:
        print('ERROR')
    return df

    
def ratio(df):
    ratio = 0
    for i in df.index:
        if df.loc[i, 'active_dummy'] == df.loc[i, 'active_predicted']:
            ratio +=1
    ratio = ratio/len(df)
    return ratio


def display_scores(scores):
    print('Scores: ', scores)
    print('Mean: ', scores.mean())
    print('Standard deviation: ', scores.std())



### Erstellt die Daten: ###
def create_data1(df):
    df = df.query('Studienjahr == 1')[features_year1].dropna().reset_index(drop = True).copy(deep = True)
    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.1, random_state = 42)
    for train_index, test_index in split.split(df, df['active_dummy']):
        df_train = df.loc[train_index, :]
        df_test = df.loc[test_index, :]
    df_train_copy = df_train.copy(deep = True)
    y_train = list(df_train['ECTS_year'])
    y_test = list(df_test['ECTS_year'])
    df_train = scaler.fit_transform(df_train.drop(['ECTS_year', 'active_dummy'], axis = 1))
    df_test = scaler.fit_transform(df_test.drop(['ECTS_year', 'active_dummy'], axis = 1))
    return df_train, y_train, df_test, y_test, df_train_copy

def create_data2(df):
    df = df.query('Studienjahr > 1')[features_years].dropna().reset_index(drop = True).copy(deep = True)
    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.1, random_state = 42)
    for train_index, test_index in split.split(df, df['active_dummy']):
        df_train = df.loc[train_index, :]
        df_test = df.loc[test_index, :]
    df_train_copy = df_train.copy(deep = True)
    y_train = list(df_train['ECTS_year'])
    y_test = list(df_test['ECTS_year'])
    df_train = scaler.fit_transform(df_train.drop(['ECTS_year', 'active_dummy'], axis = 1))
    df_test = scaler.fit_transform(df_test.drop(['ECTS_year', 'active_dummy'], axis = 1))
    return df_train, y_train, df_test, y_test, df_train_copy




### Trainiert die unterschiedlichen Modelle ###
def training_regression(df_train, y_train, df_train_copy, model): # df_train is already scaled!
    model = sklearn.base.clone(model)
    model.fit(df_train, y_train)
    
    predictions = cross_val_predict(model, df_train, y_train, cv=3)
    rmse = np.sqrt(mean_squared_error(y_train, predictions))
    mae = mean_absolute_error(y_train, predictions)
    R_squared = r2_score(y_train, predictions)
    
    print('RMSE = ' + str(rmse))
    print('MAE = ' + str(mae))
    print('R2_score = ' + str(R_squared))
    print('----------')
    df_new = active(df_train_copy, predictions)
    print('Ratio = ' + str(ratio(df_new)))
    scores = cross_val_score(model, df_train, y_train, scoring = 'neg_mean_squared_error', cv = 5)
    rmse_scores = np.sqrt(-scores)
    display_scores(rmse_scores)
    print('##########')
    
    return model

##################################################################################################################

### Generieren der Daten ###

# Year 1 :
df1_train, y1_train, df1_test, y1_test, df1_train_copy = create_data1(df)

# Year >= 2:
df2_train, y2_train, df2_test, y2_test, df2_train_copy = create_data2(df)


###################################################################################################################

### Lineare Regression ###
print('Linear Regression')
lin_reg = LinearRegression()

# Year 1:

reg1 = training_regression(df1_train, y1_train, df1_train_copy, lin_reg)

# Year >= 2:

reg2 = training_regression(df2_train, y2_train, df2_train_copy, lin_reg)

print('')




### Support Vector Machine ###
# Hyperparameters:
# kernel: linear, polynomial, gaussian or sigmoid. Uses the kernel trick to compute additional features.

# gamma: if the model ist underfitting I should increase it. If the model is overfitting I should decrease it.
 
# C: is a parameter which controls the margin violations. Higher more violations, but generalizes better.

# epsilion (only for regression): controls the width of the "street". Similar to C. However, I don't get the difference. 

print('SVM')
svm_reg = SVR(kernel = 'rbf', gamma = 10, C = 100, epsilon = 5)

# Year 1:
svm1 = training_regression(df1_train, y1_train, df1_train_copy, svm_reg)

# Year >=2:
svm2 = training_regression(df2_train, y2_train, df2_train_copy, svm_reg)
print('')








### Random Forest ###
# Hyperparameters:
# n_estimators: number of Decsiontrees.
# max_depths: maximal depth of each tree.
# max_leaf_nodes: controls the depths of the trees.
# criterion: mse or msa. function to measure the quality of the split.
# max_samples: (bootstrap = True) How many samples from df_train are drawn to train each regressor.

print('Random Forest')
forest_reg = RandomForestRegressor(
#             n_estimators = 500, max_depth = 150, max_leaf_nodes = 100, criterion = 'mae',
#             max_samples = 500
        )
# Year 1:
forest1 = training_regression(df1_train, y1_train, df1_train_copy, forest_reg)

# Year >= 2:
forest2 = training_regression(df2_train, y2_train, df2_train_copy, forest_reg)
print('')

#######################################################################################################################










### Training der KNN ###

print('KNN')
def training_ann_regression(df_train, y_train, df_train_copy):
    model = keras.models.Sequential([
        keras.layers.Dense(60, activation = 'relu', input_shape = df_train.shape[1:]),
        keras.layers.Dense(40, activation = 'relu'),
        keras.layers.Dense(20, activation = 'relu'),
        keras.layers.Dense(1, activation = 'relu')
    ])
    model.compile(loss = 'huber', optimizer = 'sgd')
    history = model.fit(df_train, np.array(y_train), epochs = 35, validation_split = 0.1)
    
    predictions = model.predict(df_train)
    rmse = np.sqrt(mean_squared_error(y_train, predictions))
    mae = mean_absolute_error(y_train, predictions)
    R_squared = r2_score(y_train, predictions)
    
    print('RMSE = ' + str(rmse))
    print('MAE = ' + str(mae))
    print('R2_score = ' + str(R_squared))
    print('----------')
    df_new = active(df_train_copy, predictions)
    print('Ratio = ' + str(ratio(df_new)))
    print('##########')
    
    return model, history

### Jahr 1 ### 

ann1, history1 = training_ann_regression(df1_train, y1_train, df1_train_copy)

pd.DataFrame(history1.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0, 25)
plt.show()


### Jahr 2 ###

ann2, history2 = training_ann_regression(df2_train, y2_train, df2_train_copy)

pd.DataFrame(history2.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0, 25)
plt.show()