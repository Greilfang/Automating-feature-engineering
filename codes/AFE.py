import numpy as np
from sklearn.neural_network import MLPClassifier
from utils.ReadRecords import read_data_features
from utils.transformation import transformations
import random
# 得到这是第几个bin
def get_bin_index(Val,Supr,Infr,Blank):
    return int((Val-Infr)/Blank)

def getScale(feature,BinNum):
    return None
'''
in:整个数据集的data
'''
def generate_binary_training_samples(OriginalSet,BinNum,SampleNum):
    UsefulTag,UselessTag = [1,0],[0,1]
    col_num = OriginalSet['data'].shape[1]
    f1,f2 = random.sample([i for i in range(col_num)],2)
    QuantifiedSketchVector=np.zeros((BinNum,2))
    QuantifiedSketchVector[:,0]=getScale(OriginalSet['data'][:,f1],BinNum)
    QuantifiedSketchVector[:,1]=getScale(OriginalSet['data'][:,f2],BinNum)
    ClassNum = len(np.unique(OriginalSet['target']))
    for index in range(SampleNum):
        pass


def generate_unary_training_samples(OriginalSet,BinNum,SampleNum):
    pass




hyparams = {
    'bin_num':10
}

feature_map=read_data_features('./utils/CleanedData.csv')
# 加载对应的transformation集合
Transformations=transformations()
Transformations.load_transformations()
#TrainSet = generate_training_samples(feature_map,hyparams['bin_num'])
