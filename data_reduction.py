#############################################################
#
# Data reduction script
# =====================
#
# The set of functions below implement the
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
#    value found. (num_group1 and num_group2 columns
#    have been dropped)
#
# 5. Dates have been made categorical (except for those in
#    the train_base.csv file. 
#
# WARNINGS
# ========
#
# 1. This script appears complete and will generate a
#    single dataframe and a csv file ~1.8GB in size called
#    master_data_file.csv
#
# 2. Polars is critical. It is significantly faster than
#    pandas and will make a huge different in running time.
#
# 3. Make sure to back-up the original data in case I
#    have introduced any bugs.
#
#############################################################


#############################################################
# Importing necessary modules
#############################################################
import numpy as np
import pandas as pd
import polars as pl
import time
#############################################################


#############################################################
# Importing file names and column names from another file
#############################################################
from features_dictionary import *
#############################################################


#############################################################
# Path to training data
#############################################################
path_train = 'csv_files/train/'
#############################################################


#############################################################
# Function to replace categorical variables by int
#############################################################
def remove_strings(files):
    print('REMOVING STRINGS')
    for i,file in enumerate(files): 			            # reading every file
        df = pd.read_csv('csv_files/train/' + file)         # importing data into a dataframe
    
        for column in df.columns:			                # step through each column in the dataframe
            if df[column].dtype == 'object' and column[-1] != 'D':   # coversion to categorical for non-date columns
                df[column] = pd.factorize(df[column])[0].astype(np.int64) 
            elif df[column].dtype == 'object' and column[-1] == 'D': # coversion to categorical for date columns
                df[column].fillna('1899-12-31', inplace=True)
                df[column] = pd.to_datetime(df[column], format='%Y-%m-%d')
                df[column] = (df[column] - pd.Timestamp('1900-01-01 00:00:00')).apply(lambda x: x.days).copy()
        df.to_csv('csv_files/train/' + file, index = False) # write new dataframe to file
        
        print(f' {i+1}/{len(files)} ' + file)               # print progress
    print('=================')
#############################################################


#############################################################
# Function to convert numbers to categorical variables by 
# quartile.
#############################################################
def float_to_cat(files):
    print('CONVERTING FLOAT COLUMNS TO CATEGORICAL')
    for i,file in enumerate(files):		                    # reading every file
        df = pd.read_csv('csv_files/train/' + file)         # importing data into a dataframe
        
        for column in df.columns:                           # step through each column in the dataframe
            if df[column].dtype == 'float64':               # check if the column is 'float64' type to be changed
                _25 = df[column].describe()[4]              # 1st quartile boundary
                _50 = df[column].describe()[5]              # 2nd
                _75 = df[column].describe()[6]              # 3rd
                A = (df[column] >= _75) * 3                 # assigning variables by quartile
                B = ((df[column] >= _50) & (df[column] < _75)) * 2
                C = ((df[column] >= _25) & (df[column] < _50)) * 1
                D = (df[column] < _25) * 0
                E = (df[column].isnull()) * -1              # NaN encoded as -1
                df[column] = (A+B+C+D+E).copy().astype(np.int64)
                
        df.to_csv('csv_files/train/' + file, index = False) # write new dataframe to file
        print(f' {i+1}/{len(files)} ' + file)               # print progress
    print('=================')
#############################################################


#############################################################
# Function to aggregate columns that have num_group1 or
# num_group2. The aggregation function simply takes whatever
# category appears the most.
#############################################################
def aggregate(files):
    print('AGGREGATING COLUMNS')
    for i,file in enumerate(files):                         # reading every file
        df = pd.read_csv(path_train + file, nrows=0).columns.tolist() # read only the column names
    
        if 'num_group1' in df:                              # read in full file if num_group1 appears
            df = pl.read_csv(path_train + file)
            data = df.group_by('case_id').agg(pl.col(df.columns[1]).map_elements(lambda x : x.value_counts(sort=True)[0,0])) # create dataframe that will aggregate by most frequent category

            for column in df.columns[2:]:
                if column not in ['num_group1', 'num_group2']:
                    df2 = df.group_by('case_id').agg(pl.col(column).map_elements(lambda x : x.value_counts(sort=True)[0,0])) # aggregate remaining columns by most frequent category
                    data = data.join(df2, on = 'case_id')   # join newly created dataframe to the main one     
        
            data.write_csv(path_train + file)               # write new dataframe to file
        
        print(f'{i+1}/{len(files)}')                        # print progress
    print('=================')
#############################################################


#############################################################
# Function that drops num_group1 and num_group2 columns.
#############################################################
def drop_num(files):
    print('DROPPING num_group1 AND num_group2 COLUMNS')
    for i,file in enumerate(files):                         # reading every file
        df = pl.read_csv(path_train + file)
    
        if 'num_group2' in df.columns:
            df = df.drop('num_group1')                      # drop num_group1
        if 'num_group2' in df.columns:
            df = df.drop('num_group2')                      # drop num_group2

        df.write_csv(path_train + file)    

        print(f'{i+1}/{len(files)} ' + file)                # print progress
    print('=================')           
#############################################################


#############################################################
# Function that reads in all files and attaches them ton a
# single master table.
#############################################################
 def attach(columns, dataframes, df):
    print('ATTACHING COLUMNS')
    for i,column in enumerate(columns):                     # go over every column that is not 'case_id'
        if column != 'case_id':
            DF1 = pl.DataFrame()                            # create an empty dataframe with two columns, 'case_id' and the current column
            DF1.with_columns(pl.lit(None).alias('case_id'))
            DF1.with_columns(pl.lit(None).alias(column))
            for d in dataframes:                            # check every dataframe that has the same column name and concatenate the column to DF1
                if column in d.columns:                     # then aggregate by most frequent category as above
                    DF1 = pl.concat([DF1, d[['case_id', column]]])
        DF1.group_by('case_id').agg(pl.col(column).map_elements(lambda x : x.value_counts(sort=True)[0,0]))
        df = df.join(DF1, on = 'case_id', how = 'left')     # join new column containing the aggregate of the same category spread over all dataframes to the master dataframe
        print(f'{i+1}/{len(columns)}')                      # print progress
    print('=================')                          
    return df                                               # returns master dataframe
#############################################################


#############################################################
# Main function that modifies all by the train_base.csv file
#############################################################
def modify_data(files,columns):
    a = time.time()
    remove_strings(files[1:])                               # modifies strings in all but train_base.csv file
    float_to_cat(files[1:])                                 # float to categorical in all but train_base.csv file
    aggregate(files[1:])				                    # aggregates as mentioned above in all but train_base.csv file
    drop_num(files[1:])                                     # drops as mentioned above in all but train_base.csv file
    for column in df.columns:                               # casts float columns as int
        if df[column].dtype == 'float64':
            df[column] = df[column].astype('int64').copy()
    df = pl.read_csv(path_train + files[0])                 # read in train_base.csv file
    D = [pl.read_csv(path_train + file) for file in files]  # read all over files in a list of dataframes
    df = attach(columns, D, df)                             # begin attaching columns from all other dataframes to the main one from train_base.csv
    df = df.to_pandas()                                     # convert the dataframe to pandas
    df['date_decision'] = pd.to_datetime(df['date_decision'], format='%Y-%m-%d') # convert datetime column
    df.fillna(-1, inplace= True)                            # replace all NaN by -1 that were introduced in the merging process
    b = time.time()
    print('DONE!')
    print(b-a)
    return df		                                        # return master dataframe
#############################################################


#############################################################
# Apply all of the above modifications to the dataset
#############################################################
df = modify_data(FILES,COLUMNS)
#############################################################


#############################################################
# Write final table to 'master_data_file.csv'
#############################################################
df.to_csv('master_data_file.csv', index=False)
#############################################################
