import numpy as np
from sklearn.neural_network import MLPClassifier
from utils.ReadRecords import read_data_features,save_transformations,convert
from utils.transformation import transformations
from sklearn.ensemble import RandomForestClassifier
import copy
import pickle

# 设置超参数
hyparams = {
    'bin_num': 10,
    'improvement': 0.01,
    'unary_sample_num': 50,
    'binary_sample_num': 150,
    'raw_root':'./raw'
}

if __name__ == "__main__":
    root=hyparams['raw_root']
    OriginalSets = convert(root)

    # 加载对应的transformation集合
    Transformations = transformations()
    Transformations.upload()

    # 产生QSA数据集
    Transformations.generate_training_samples(OriginalSets=OriginalSets,hyparams=hyparams)
    save_transformations(Transformations, path='./model/transformations')
