import pickle
import numpy as np
from utils.ReadRecords import load_transformations,save_transformations
from utils.transformation import transformations
from sklearn.neural_network import MLPClassifier


#Transformations=load_transformations('./model/transformations')
#用在AFE.py文件里储存的数据集继续训练

def training(Transformations):
    TrainingDatas=Transformations.DataSets    
    # 训练 一元变化MLP
    for trans_name,unary_MLP in Transformations.unary_MLPs.items():
        TrainingData = TrainingDatas[trans_name]
        assert(isinstance(unary_MLP,MLPClassifier))
        unary_MLP.fit(TrainingData['data'],TrainingData['target'])
        print(trans_name,' training finish ')
    
    print('-----------------unary and binary-------------------')
    # 训练 二元变化MLP
    for trans_name,binary_MLP in Transformations.binary_MLPs.items():
        TrainingData = TrainingDatas[trans_name]
        assert(isinstance(binary_MLP,MLPClassifier))
        binary_MLP.fit(TrainingData['data'],TrainingData['target'])
        print(trans_name,' training finish ')


    

if __name__ == "__main__":
    Transformations = load_transformations('model/transformations')
    training(Transformations)
    save_transformations(Transformations,'model/transformations')
