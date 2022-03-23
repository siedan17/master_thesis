
### Import of used libraries ###

import sys
import os
import random as rd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_path = '../raw_data'
cwd = os.path.dirname(__file__)
raw_data_folder = os.path.abspath(os.path.join(cwd, data_path))


def load_df(name):
    return pd.read_csv(os.path.join(raw_data_folder, name))

df = load_df('adapted_data.csv').drop(['Unnamed: 0'], axis =1)

# copy, for doubled usage afterwards.
df1 = df.copy(deep = True)


features = ['year','Studienjahr', 'active_year_before', 'geschlecht', 
            'status_year_before', 'country', 'school', 'subject', 'active_dummy', 
            'status_key']

###################################################################################################

### Helper Functions ###

def active_year_before(df): 
    list_studienjahre = list(df['year'].unique())
    list_studienjahre.sort()
    for i in list_studienjahre[1:]:
        df_year = df.query('year == @i')
        df_year0 = df.query('year == (@i-1)')
        for num in df_year.index:
            if df_year.loc[num, 'Studienjahr'] > 1:
                matrikel = df_year.loc[num, 'matrikel_num']
                if (num-1) in df_year0.index and df_year0.loc[(num-1), 'matrikel_num'] == matrikel:
                    df.loc[num, 'active_year_before'] = df.loc[(num-1), 'active_dummy']
                    df.loc[num, 'status_year_before'] = df.loc[(num-1), 'status_key']
            
    return df

def country(df):
    for i in range(len(df)):
        if df.loc[i, 'styria_dummy'] == 1:
            df.loc[i, 'country'] = 0
        elif df.loc[i, 'not_styria_dummy'] == 1:
            df.loc[i, 'country'] = 1
        elif df.loc[i, 'germany_dummy'] == 1:
            df.loc[i, 'country'] = 2
        else:
            df.loc[i, 'country'] = 3
    return df

def school(df):
    for i in range(len(df)):
        if df.loc[i, 'AHS_dummy'] == 1:
            df.loc[i, 'school'] = 0
        elif df.loc[i, 'BHS_dummy'] == 1:
            df.loc[i, 'school'] = 1
        else:
            df.loc[i, 'school'] = 2
    return df

def subject(df):
    for i  in range(len(df)):
        if df.loc[i, 'jus_dummy'] == 1:
            df.loc[i, 'subject'] = 0
        elif df.loc[i, 'bwl_dummy'] == 1:
            df.loc[i, 'subject'] = 1
        else:
            df.loc[i, 'subject'] = 2
    return df


### Applying them ###

active_year_before(df)
country(df)
school(df)
subject(df)

df_working = df[features]

### Helper Function for creating all combinations ###

def create_combinations():
    return_list = []
    for a in range(2):
        for b in range(3):
            for c in range(4):
                for d in range(3):
                    return_list.append([a,b,c,d])
    return return_list

numerical_combinations = create_combinations()
print(len(numerical_combinations))

def split(df, combinations):
    return_list = []
    for i in combinations:
        geschlecht = i[0]
        subject = i[1]
        country = i[2]
        school = i[3]
        
        df_append = df.query('geschlecht == @geschlecht and subject == @subject and country == @country and school == @school')
        return_list.append((i, df_append))
    return return_list
        
### Studienjahr hier festlegen !!!!! ###
### Aktuell Studienjahr 2 ###
df_list = split(df_working.query('Studienjahr == 2'), numerical_combinations)


### Helper Functions for calculating probabilities ###


def prob_year(df): # returns a list with probabilities
    a = len(df.query("active_dummy == 1 and status_key != 'I'"))
    b = len(df.query("active_dummy == 0 and status_key != 'I'"))
    c = len(df.query("active_dummy == 1 and status_key == 'I'"))
    d = len(df.query("active_dummy == 0 and status_key == 'I'"))    
    
    denominator = len(df)
    if denominator > 0:
        return [round(a/denominator, 2), round(b/denominator,2), round(c/denominator,2),  round(d/denominator, 2)]
    else:
        return "Zero"
    
def print_matrix(df):
    df_active = df.query('active_year_before == 1')
    df_inactive = df.query('active_year_before == 0')

    print(prob_year(df_active))
    print(prob_year(df_inactive))
    print('---------')




######################################################################################################################



### Calculating probability for 1st Year for the first class (0,0,0,0)
# This is: weiblich, Steiermark, JUS, AHS Vorbildung.

# Could be solved more elegant!!!

print('--------')
print(prob_year(df_working.query('Studienjahr == 1 and subject == 0 and country == 0 and geschlecht == 0 and school == 0')))





### Calculating all probabilities for All classes for year 2 ###


for i in df_list:
    print(i[0]) # Welche Kombination es ist.
    print(len(i[1])) # Wie viele Eintrge die Klasse hat.
    print_matrix(i[1]) # Wahrscheinlichkeitsvektoren wie im Bericht beschrieben. 


###############################################################################################################################