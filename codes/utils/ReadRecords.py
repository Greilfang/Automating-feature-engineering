import numpy as np 
import pandas as pd
import pickle
def read_data_features(path):
    dataset=pd.read_csv(path)
    print(dataset.columns)
    feature_map={}
    values=dataset.values
    cols = values.shape[1]
    for col in range(cols):
        feature_map[dataset.columns[col]]=values[:,col]
    return feature_map

def save_transformations(Transformations, path):
    with open(path,'wb') as file:
        pickle.dump(Transformations, file)


def load_transformations(path):
    with open(path,'rb') as file:
        Transformations = pickle.load(file)
    return Transformations


