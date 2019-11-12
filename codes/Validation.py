from utils.ReadRecords import load_transformations,convert
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
hyparams={
    'load_path':'./model/transformations',
    'data_path':'./utils/CleanedData.csv'
}

def getTargetDataSet(path):
    df = pd.read_csv(hyparams['data_path'])

if __name__ == "__main__":
    # 加载模型
    model = load_transformations(hyparams['load_path'])
    # 加载数据集
    TargetSet = getTargetDataSet(hyparams['data_path'])
    model.predict(TargetSet)

