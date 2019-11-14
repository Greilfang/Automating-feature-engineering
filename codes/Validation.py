from utils.ReadRecords import load_transformations,convert
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
hyparams={
    'load_path':'./model/transformations',
    'data_path':'./utils/CreditData.csv',
    'root_path':'../ValSet/',
    'threshold':0.9
}


def getTargetDataSet(path):
    df = pd.read_csv(path)
    dataset={
        'data':df.values[:,:-1],
        'target':df.values[:,-1].astype('int'),
        'attributes':list(df.columns)[:-1],
        'name':'CleanedData'
    }
    return dataset


if __name__ == "__main__":
    # 加载模型
    model = load_transformations(hyparams['load_path'])
    # 加载数据集
    TargetSet = getTargetDataSet(hyparams['data_path'])
    threshold = hyparams['threshold']
    model.predict(TargetSet,5 ,threshold)