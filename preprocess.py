import os
import pandas as pd
import numpy as np
from sklearn import preprocessing


path = os.getcwd()
path_to_datasets = r'/materials/TrafficLabelling'
all_files = os.listdir(path + path_to_datasets)    
csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))

for i in range(len(csv_files)):
    df = pd.read_csv(csv_files[i])
    df.columns = df.columns.str.strip()
    main_features = set(df.columns)
    for feature in main_features:
        if df[feature].isnull().sum() > 0:
            df[feature].fillna(0, inplace = True)
    numeric_columns = set(df.select_dtypes(include=np.number).columns)

    for column in numeric_columns:
        if np.isin([-np.inf, np.inf], df[feature]).any():
            df[column].replace([np.inf, -np.inf], -1, inplace=True)
    non_numeric_columns = main_features.difference(numeric_columns)

    for column in non_numeric_columns:
        df[column].str.replace('–','-')

    labelencoder = preprocessing.LabelEncoder()
    non_numeric_columns.remove('Label')
    non_numeric_columns = main_features.difference(numeric_columns)

    for column in non_numeric_columns:
        df[column] = labelencoder.fit_transform(df[column])
    for i in range(len(csv_files)):
        df.to_csv('{i}.csv'.format(i))
