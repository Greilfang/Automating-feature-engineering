from utils.ReadRecords import read_data_features,save_transformations,convert,load_transformations
from collections import Counter
import pandas as pd
import numpy as np
def printOutTrainingDatas(Transformations):
    for name,dataset in Transformations.DataSets.items():
        print('-----------------------')
        targets=dataset['target']
        stats=pd.value_counts(list(targets[:,0]))
        print('name:',name)
        print('-------------------------')
        print(stats)
        #pad={'name':name,'useful':stats[1],'useless':stats[0]}
        #print(pad)


if __name__ == "__main__":
    Transformations = load_transformations('model/transformations')
    printOutTrainingDatas(Transformations)


