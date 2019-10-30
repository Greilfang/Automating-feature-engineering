import numpy as np 
import pandas as pd

def read_data_features(path):
    dataset=pd.read_csv(path)
    print(dataset.columns)
    feature_map={}
    values=dataset.values
    cols = values.shape[1]
    for col in range(cols):
        feature_map[dataset.columns[col]]=values[:,col]
    return feature_map
