import os
import pandas as pd
import numpy as np
from unidecode import unidecode
from sklearn import preprocessing


path = os.getcwd()
path_to_datasets = r'/materials/TrafficLabelling'
all_files = os.listdir(path + path_to_datasets) 
csv_files = list(filter(lambda f: f.endswith('.pcap_ISCX.csv'), all_files))
full_dataset = pd.DataFrame([])

for i in range(len(csv_files)):
    with open(path + path_to_datasets + r'/' + csv_files[i], encoding="utf8", errors='ignore') as file:
        df = pd.read_csv(file, low_memory=False)
    df.columns = df.columns.str.strip()
    main_features = set(df.columns)

    numeric_columns = set(df.select_dtypes(include=np.number).columns)
    for feature in numeric_columns:
        if df[feature].isnull().sum() > 0:
            df[feature].fillna(0, inplace = True)
    numeric_columns = set(df.select_dtypes(include=np.number).columns)

    for column in numeric_columns:
        if np.isin([-np.inf, np.inf], df[column]).any():
            df[column].replace([np.inf, -np.inf], -1, inplace=True)
    non_numeric_columns = main_features.difference(numeric_columns)
    df = df.dropna()

    for column in non_numeric_columns:
        df[column].str.replace('–', '-') 
        #df[column] = df[column].apply(unidecode)

    labelencoder = preprocessing.LabelEncoder()
    non_numeric_columns.remove('Label')
    non_numeric_columns = main_features.difference(numeric_columns)
    print(df[non_numeric_columns].dtypes)

    for column in non_numeric_columns:
        df[column] = labelencoder.fit_transform(df[column])

    full_dataset = pd.concat([full_dataset, df])
    print(csv_files[i], ' preprocessed\n')

full_dataset.to_csv('full_dataset.csv')
