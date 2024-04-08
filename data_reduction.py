#############################################################
#
# Data reduction script
# =====================
#
# The following set of functions implement the
# following assumptions in order to reduce the
# size of the dataset:
#
# 1. All categorical variables will be converted to
#    int64 (except the datetime on train_base.csv).
#
# 2. All float64 columns will be converted to int64
#    categories according to their quartiles.
#
# 3. NaN values will be encoded as -1 and not dropped.
#
# 4. For files containing num_group1 or num_group2,
#    aggregation is done by taking the most common
#    value found.
#
# WARNINGS
# ========
#
# 1. This script is incomplete, it is lacking a way to 
#    merge all files into a single master dataframe.
#    This is a work in progress.
#
# 2. Polars is critical. It is significantly faster than
#    pandas and will make a huge different in running time.
#
# 3. Make sure to back-up the original data in case I
#    have introduced any bugs in the reduction() function.
#
#############################################################

import numpy as np
import pandas as pd
import polars as pl
import datetime

from features_dictionary import *
f_2_f = features_to_files

path_train = 'csv_files/train/'

def remove_strings(files):
    
    for i,file in enumerate(files):
        df = pd.read_csv('csv_files/train/' + file)
    
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = pd.factorize(df[column])[0]
        df.to_csv('csv_files/train/' + file, index = False)
        
        print(f' {i+1}/{len(files)} ' + file)
		
def float_to_cat(files):
    
    for i,file in enumerate(files):
        df = pd.read_csv('csv_files/train/' + file)
        
        for column in df.columns:
            if df[column].dtype == 'float64':
                _25 = df[column].describe()[4]
                _50 = df[column].describe()[5]
                _75 = df[column].describe()[6]
                A = (df[column] >= _75) * 3
                B = ((df[column] >= _50) & (df[column] < _75)) * 2
                C = ((df[column] >= _25) & (df[column] < _50)) * 1
                D = (df[column] < _25) * 0
                E = (df[column].isnull()) * -1
                df[column] = (A+B+C+D+E).copy().astype(np.int64)
                
        df.to_csv('csv_files/train/' + file, index = False)
        print(f' {i+1}/{len(files)} ' + file)      
		
def aggregate(files):
    for i,file in enumerate(files):
        df = pd.read_csv(path_train + file, nrows=0).columns.tolist()
    
        if 'num_group1' in df:
            df = pl.read_csv(path_train + file)
            data = df.group_by('case_id').agg(pl.col(df.columns[1]).map_elements(lambda x : x.value_counts()[0,1]))

            for column in df.columns[2:]:
                df2 = df.group_by('case_id').agg(pl.col(column).map_elements(lambda x : x.value_counts()[0,1]))
                data = data.join(df2, on = 'case_id')
        
            data.write_csv(path_train + file)    
        
        print(f'{i+1}/{len(FILES)}')
	
def reduction():	
	remove_strings(FILES[1:])
	float_to_cat(FILES[1:])
	aggregate(FILES[1:])
	df = pd.read_csv(path_train + FILES[0])
	df['date_decision'] = pd.to_datetime(df['date_decision'], format='%Y-%m-%d')
	D = [pd.read_csv(path_train + file) for file in FILES[1:]]