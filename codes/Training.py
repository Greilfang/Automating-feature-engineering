import pickle
import numpy as np
from utils.ReadRecords import load_transformations,save_transformations
from utils.transformation import transformations
from sklearn.neural_network import MLPClassifier


#Transformations=load_transformations('./model/transformations')
#用在AFE.py文件里储存的数据集继续训练

def training(Transformations):
    TrainingDatas=Transformations.DataSets
    #对数据集内数据进行重新设置维数使其输入mlp
    for trans_name,trans_set in TrainingDatas.keys(),TrainingDatas.values():
        trans_set['data']=list(map(lambda x: x.flatten(),trans_set['data']))
        trans_set['data']=np.array(trans_set['data'])
    
    
    # 训练 一元变化MLP
    for trans_name,unary_MLP in Transformations.unary_MLPs:
        TrainingData = TrainingDatas['trans_name']
        assert(isinstance(unary_MLP,MLPClassifier))
        unary_MLP.fit(TrainingData['data'],TrainingData['target'])
    
    
    # 训练 二元变化MLP
    for trans_name,binary_MLP in Transformations.binary_MLPs:
        TrainingData = TrainingDatas['trans_name']
        assert(isinstance(unary_MLP,MLPClassifier))
        binary_MLP.fit(TrainingData['data'],TrainingData['target'])


    

if __name__ == "__main__":
    a = np.array([[1,2,3,4,5],[6,7,8,9,10]])
    b = np.array([[-1,-2,-3,-4,-5],[-6,-7,-8,-9,-10]])
    c=[]
    c.append(a)
    c.append(b)
    c = list(map(lambda x: x.flatten(),c))
    k =np.array(c)
    print(k)