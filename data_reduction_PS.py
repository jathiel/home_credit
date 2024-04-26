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
import os.path

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
path_train_read = 'data/csv_files/train/'
path_train_save = 'data/csv_files/cleaned/'


#############################################################


#############################################################
# Function to replace categorical variables by int
#############################################################
def remove_strings(files, path, check_modified=True):
    print('REMOVING STRINGS')
    for i, file in enumerate(files):  # reading every file
        if check_modified:
            if os.path.exists(path_train_save + file):
                continue
        df_string = pd.read_csv(path + file)  # importing data into a dataframe

        for column in df_string.columns:  # step through each column in the dataframe
            if df_string[column].dtype == 'object' and column[
                -1] != 'D':  # coversion to categorical for non-date columns
                df_string[column] = pd.factorize(df_string[column])[0].astype(np.int64)
            elif df_string[column].dtype == 'object' and column[-1] == 'D':  # coversion to categorical for date columns
                df_string[column].fillna('1899-12-31', inplace=True)
                df_string[column] = pd.to_datetime(df_string[column], format='%Y-%m-%d')
                df_string[column] = (df_string[column] - pd.Timestamp('1900-01-01 00:00:00')).apply(
                    lambda x: x.days).copy()
        df_string.to_csv(path_train_save + file, index=False)  # write new dataframe to file

        print(f' {i + 1}/{len(files)} ' + file)  # print progress
    print('=================')


#############################################################


#############################################################
# Function to convert numbers to categorical variables by 
# quartile.
#############################################################
def float_to_cat(files, path, check_modified=True):
    print('CONVERTING FLOAT COLUMNS TO CATEGORICAL')
    for i, file in enumerate(files):  # reading every file
        if check_modified:
            if os.path.exists(path_train_save + file):
                continue
        df_f2c = pd.read_csv(path_train_save + file)  # importing data into a dataframe

        for column in df_f2c.columns:  # step through each column in the dataframe
            if df_f2c[column].dtype == 'float64':  # check if the column is 'float64' type to be changed
                _25 = df_f2c[column].describe()[4]  # 1st quartile boundary
                _50 = df_f2c[column].describe()[5]  # 2nd
                _75 = df_f2c[column].describe()[6]  # 3rd
                A = (df_f2c[column] >= _75) * 3  # assigning variables by quartile
                B = ((df_f2c[column] >= _50) & (df_f2c[column] < _75)) * 2
                C = ((df_f2c[column] >= _25) & (df_f2c[column] < _50)) * 1
                D = (df_f2c[column] < _25) * 0
                E = (df_f2c[column].isnull()) * -1  # NaN encoded as -1
                df_f2c[column] = (A + B + C + D + E).copy().astype(np.int64)

        df_f2c.to_csv(path + file, index=False)  # write new dataframe to file
        print(f' {i + 1}/{len(files)} ' + file)  # print progress
    print('=================')


#############################################################


#############################################################
# Function to aggregate columns that have num_group1 or
# num_group2. The aggregation function simply takes whatever
# category appears the most.
#############################################################
def aggregate(files, path, check_modified):
    print('AGGREGATING COLUMNS')
    for i, file in enumerate(files):  # reading every file
        if check_modified:
            if os.path.exists(path_train_save + file):
                continue
        df_aggregate = pd.read_csv(path + file, nrows=0).columns.tolist()  # read only the column names

        if 'num_group1' in df_aggregate:  # read in full file if num_group1 appears
            df_aggregate = pl.read_csv(path + file)
            data = df_aggregate.group_by('case_id').agg(
                pl.col(df_aggregate.columns[1]).map_elements(lambda x: x.value_counts(sort=True)[
                    0, 0]))  # create dataframe that will aggregate by most frequent category

            for column in df_aggregate.columns[2:]:
                if column not in ['num_group1', 'num_group2']:
                    df2 = df_aggregate.group_by('case_id').agg(
                        pl.col(column).map_elements(lambda x: x.value_counts(sort=True)[
                            0, 0]))  # aggregate remaining columns by most frequent category
                    data = data.join(df2, on='case_id')  # join newly created dataframe to the main one

            data.write_csv(path + file)  # write new dataframe to file

        print(f'{i + 1}/{len(files)}')  # print progress
    print('=================')


#############################################################


#############################################################
# Function that drops num_group1 and num_group2 columns.
#############################################################
def drop_num(files, path, check_modified=True):
    print('DROPPING num_group1 AND num_group2 COLUMNS')
    for i, file in enumerate(files):  # reading every file
        if check_modified:
            if os.path.exists(path_train_save + file):
                continue
        df_drop = pl.read_csv(path + file)

        if 'num_group2' in df_drop.columns:
            df_drop = df_drop.drop('num_group1')  # drop num_group1
        if 'num_group2' in df_drop.columns:
            df_drop = df_drop.drop('num_group2')  # drop num_group2

        df_drop.write_csv(path + file)

        print(f'{i + 1}/{len(files)} ' + file)  # print progress
    print('=================')


#############################################################


#############################################################
# Function that reads in all files and attaches them ton a
# single master table.
#############################################################
def attach(columns, dataframes, df):
    print('ATTACHING COLUMNS')
    df_results = pl.DataFrame()

    for i, column in enumerate(columns):  # go over every column that is not 'case_id'
        if column != 'case_id':
            # DF1 = pl.DataFrame()  # create an empty dataframe with two columns, 'case_id' and the current column
            # DF1.with_columns(pl.lit(None).alias('case_id'))
            # DF1.with_columns(pl.lit(None).alias(column))
            DF1 = pl.DataFrame({'case_id': [], column: []})

            for d in dataframes:  # check every dataframe that has the same column name and concatenate the column to DF1
                if column in d.columns:  # then aggregate by most frequent category as above
                    DF1 = pl.concat([DF1, d[['case_id', column]]])
        DF1_agg = DF1.groupby('case_id').agg(
            pl.col(column).map_elements(lambda x: x.value_counts(sort=True)[0, 0]).alias(column))
        if df_results.height == 0:  # If it's the first processed column, initialize df_results
            df_results = DF1_agg
        else:
            df_results = df_results.join(DF1_agg, on='case_id', how='left')
        # DF1.group_by('case_id').agg(pl.col(column).map_elements(lambda x: x.value_counts(sort=True)[0, 0]))
        # df = df.join(DF1, on='case_id',
        #              how='left')  # join new column containing the aggregate of the same category spread over all dataframes to the master dataframe
        print(f'{i + 1}/{len(columns)}')  # print progress
    print('=================')
    return df_results  # returns master dataframe

import polars as pl


def process_dataframe(df, column):
    return df.groupby('case_id').agg(
        pl.col(column).mode().alias(column)
    )


# def attach(columns, dataframe_paths, base_df_path):
#     print('ATTACHING COLUMNS')
#     df_base = pl.read_csv(base_df_path)  # Load base dataframe
#
#     # Initialize result dataframe with the base 'case_id'
#     df_results = df_base.select('case_id').unique()
#
#     for column in columns:
#         try:
#             if column != 'case_id':
#                 DF1 = pl.DataFrame({'case_id': [], column: []})  # Temporary dataframe to aggregate column data
#
#                 for path in dataframe_paths:
#                     df_temp = pl.scan_csv(path)  # Lazily load data
#
#                     if column in df_temp.columns:
#                         # Select only necessary columns and append to DF1
#                         df_processed = df_temp.select(['case_id', column]).collect()
#                         DF1 = pl.concat([DF1, df_processed])
#
#                 # Aggregate data by 'case_id', selecting the most frequent value per case
#                 DF1_agg = DF1.groupby('case_id').agg([
#                     (pl.col(column).mode().alias(column))
#                 ])
#
#                 # Join aggregated data back to results
#                 df_results = df_results.join(DF1_agg, on='case_id', how='left')
#         except Exception as e:
#             print(f"{type(e)}: Column {column} cannot be concatenated. Skipping...")
#     return df_results


#############################################################
# Main function that modifies all by the train_base.csv file
#############################################################
def modify_data(files, check_modified=True):
    df = pd.read_csv(path_train_read+files[0])
    df.to_csv(path_train_save + files[0])
    remove_strings(files[1:], path_train_read, check_modified)  # modifies strings in all but train_base.csv file
    float_to_cat(files[1:], path_train_save, check_modified)  # float to categorical in all but train_base.csv file
    aggregate(files[1:], path_train_save,
              check_modified)  # aggregates as mentioned above in all but train_base.csv file
    drop_num(files[1:], path_train_save, check_modified)  # drops as mentioned above in all but train_base.csv file
    # for column in df.columns:  # casts float columns as int
    #     if df[column].dtype == 'float64':
    #         df[column] = df[column].astype('int64').copy()


def combine_data(files, columns, modify=True, check_modified=True):
    a = time.time()
    if modify:
        modify_data(files, check_modified)
        path_save = path_train_save
    else:
        path_save = path_train_read
    # List of paths to your CSV files
    dataframe_paths = [path_save + file for file in files]

    # Attach columns from all dataframes to the base dataframe
    df_combine = attach(columns, dataframe_paths, path_save + files[0])
    df_combine = df_combine.to_pandas()  # convert the dataframe to pandas
    df_combine['date_decision'] = pd.to_datetime(df_combine['date_decision'],
                                                 format='%Y-%m-%d')  # convert datetime column
    df_combine.fillna(-1, inplace=True)  # replace all NaN by -1 that were introduced in the merging process
    b = time.time()
    print('DONE!')
    print(b - a)
    return df_combine  # return master dataframe


#############################################################

if __name__ == "__main__":
    modify = True

    #############################################################

    #############################################################
    # Apply all of the above modifications to the dataset
    #############################################################
    df = combine_data(FILES, COLUMNS, modify=modify)
    #############################################################

    #############################################################
    # Write final table to 'master_data_file.csv'
    #############################################################
    savepath = f"/data/csv_files/master_data_file{'' if modify else '_agg'}.csv"
    df.to_csv(savepath, index=False)
    #############################################################
