from utils.ReadRecords import load_transformations
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
hyparams={
    'load_path':'./model/transformations'
}

if __name__ == "__main__":
    # 加载模型
    model = load_transformations(hyparams['load_path'])

