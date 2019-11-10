import pickle
import numpy as np
from utils.ReadRecords import load_transformations
from utils.transformation import transformations


#Transformations=load_transformations('./model/transformations')
#用在AFE.py文件里储存的数据集继续训练

def training(Transformations):
    TrainingData=Transformations.DataSets
    #对数据集内数据进行重新设置维数使其输入mlp
    for trans_name,trans_set in TrainingData.keys(),TrainingData.values():
        trans_set['data']=list(map(lambda x: x.flatten(),trans_set['data']))
        trans_set['data']=np.array(trans_set['data'])
    

if __name__ == "__main__":
    a = np.array([[1,2,3,4,5],[6,7,8,9,10]])
    b = np.array([[-1,-2,-3,-4,-5],[-6,-7,-8,-9,-10]])
    c=[]
    c.append(a)
    c.append(b)
    c = list(map(lambda x: x.flatten(),c))
    k =np.array(c)
    print(k)