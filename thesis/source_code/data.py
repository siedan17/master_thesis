### importing libraries ###

import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



### importing data files ###

# relative path to the data folder

data_path = '../raw_data'
cwd = os.path.dirname(__file__)
raw_data_folder = os.path.abspath(os.path.join(cwd, data_path))



# labels of features, which will be used later on

columns = [
    'matrikel_num', 'Studienjahr',
    'geschlecht', 'avgECTS_sem_before',
    'ects_year_before', 'year',
    'first_exam_negative', 'AHS_dummy',
    'BHS_dummy', 'ausland_vorbildung_dummy',
    'sonstige_vorbildung_dummy', 'delayed_dummy',
    'num_parallel_studies', 'jus_dummy',
    'bwl_dummy', 'years_since_matura',
    'styria_dummy', 'not_styria_dummy',
    'germany_dummy', 'other_foreign_dummy',
    'years_since_18', 'planned_duration',
    'full_duration_sem', 'full_duration_sem_before',
    'firstGen', 'cum_ects_pos_before',
    'status_key', 'ECTS_year',
    'SWS_year', 'active_dummy',
    'subject'
    ]



# helper function for importing data. Returns a "Pandas DataFrame" object.
def load_df(name):
    return pd.read_csv(os.path.join(raw_data_folder, name))

df = load_df('bwl_pad_jus.csv').reset_index(drop = True)



### helper functions for calculating new features ###

# calculating "full_duration_sem_before" as a new column. This is the number of semesters the student has already studied.
def full_duration_sem_before(df):
    for i in range(len(df)):
        a = math.floor(df.loc[i, "full_duration_sem"])
        if a <= 1:
            df.loc[i, "full_duration_sem_before"] = 0
        else:
            if df.loc[i, "last_semester_name"][2] == 'W':
                df.loc[i, "full_duration_sem_before"] = a - 1
            else:
                df.loc[i, "full_duration_sem_before"] = a - 2
    return df



# calculating "geschlecht" categorial.
def geschlecht(df):
    for i in range(len(df)):
        a = df.loc[i, "geschlecht"]
        if a == "W":
            df.loc[i, "geschlecht"] = 0
        else:
            df.loc[i,"geschlecht"] = 1
    return df

# calculating 'cum_ects_pos_before'. Calculating the ECTS before the current year. If not, we would be using
# information, which should actually be learned afterwards.
def cum_ects_pos_before(df):
    df['cum_ects_pos_before'] = df['avgECTS_sem_before']*df['full_duration_sem_before']
    return df

        
# calculating "avgECTS_sem_before" as a new column. These are the average ECTS per semester before the current year.
def avgECTS_sem_before(df):
    for i in range(len(df)):
        cum_ects_before_pseudo = df.loc[i, 'cumulated_ects_pos'] - df.loc[i, 'ECTS_year']
        if not df.loc[i, 'full_duration_sem_before'] == 0:
            df.loc[i, 'avgECTS_sem_before'] = float(cum_ects_before_pseudo) / df.loc[i, 'full_duration_sem_before']
        else:
            df.loc[i, 'avgECTS_sem_before'] = 0
    return df 


# calculating "Studienjahr" as a new column.
def Studienjahr(df):
    for i in range(len(df)):
        df.loc[i, "Studienjahr"] = int((df.loc[i, "full_duration_sem_before"]//2) + (df.loc[i, "full_duration_sem_before"]%2) + 1)
    return df


# deleting rows with negative "years since matura".
def Matura(df):
    df.drop(df[df['years_since_matura'] < 0].index, inplace = True)
    return df



# calculating "ECTS_year_before". Only if they are available in the data.
def year(df):
    for i in df.index:
        num = df.loc[i, 'year'][:2]
        df.loc[i, 'year'] = int(num)
    return df

def ects_year_before(df):  
    list_studienjahre = list(df['year'].unique())
    list_studienjahre.sort()
    for i in list_studienjahre[1:]:
        df_year = df.query('year == @i')
        df_year0 = df.query('year == (@i-1)')
        
        for num in df_year.index:
            if df_year.loc[num, 'Studienjahr'] > 1:
                matr = df_year.loc[num, 'matrikel_num']
                if (num-1) in df_year0.index and df_year0.loc[(num-1), 'matrikel_num'] == matr:
                    df.loc[num, 'ects_year_before'] = df.loc[(num-1), 'ECTS_year']
            
    return df


# calculating one-hot-encoding for subject.
def subject(df):
    for i in range((len(df))):
        if df.loc[i, 'subject'] == 'Rechtswissenschaften':
            df.loc[i, 'jus_dummy'] = 1
            df.loc[i, 'bwl_dummy'] = 0
        elif df.loc[i, 'subject'] == 'Betriebswirtschaft':
            df.loc[i, 'jus_dummy'] = 0
            df.loc[i, 'bwl_dummy'] = 1
        else:
            df.loc[i, 'jus_dummy'] = 0
            df.loc[i, 'bwl_dummy'] = 0
    return df



### here I am actually manipulating the data with the helper functions. ###
# functions for saving manipulated DataFrame
def saving(df, name):
    df.to_csv(os.path.join(raw_data_folder, name))


def data_manipulation_pipeline(df, columns, name, save_or_return): # does manipulations to df and is able to save
    subject(df)
    geschlecht(df)
    full_duration_sem_before(df)
    avgECTS_sem_before(df)
    cum_ects_pos_before(df)
    Studienjahr(df)
    Matura(df)
    year(df)
    ects_year_before(df)
    
    df = df[columns] # ordering of columns

    if save_or_return:
        saving(df, name) 
    else:
        return df

# calling the pipeline  
data_manipulation_pipeline(df, columns, 'adapted_data.csv', True)

