import numpy as np 
import pandas as pd
import pickle
from sklearn.utils import shuffle
import os
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


def convert(root):
    OriginalSets=[]
    csvs =os.listdir(root)
    print(csvs)
    print('-----------------')
    for csv in csvs:
        print(csv)
        path = root +'/'+csv
        if os.path.isfile(path) and os.path.splitext(path)[-1] == '.csv':
            print('csv_name:',path)
            name = os.path.splitext(csv)[0]
            df=pd.read_csv(path)
            shuffle(df)
            OriginalSet = {
                'data':df.values[:,0:-1],
                'target':df.values[:,-1].astype('int'),
                'name':name,
                'attributes':list(df.columns)
                }
            OriginalSets.append(OriginalSet)
    return OriginalSets

# print(read_data_features('../../raw/hmeq.csv'))
# convert('../../raw/')