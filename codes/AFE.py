import numpy as np
from sklearn.neural_network import MLPClassifier
from utils.ReadRecords import read_data_features
from utils.transformation import transformations,TransformationDataSets
from sklearn.ensemble import RandomForestClassifier
import random
import copy




# 得到分位数分布
def getSketch(feature_t,BinNum,cls_num,classes):
    QuantifiedSketchVector=np.zeros((BinNum,cls_num))
    Supr,Infr=max(feature_t),min(feature_t)
    Blank = Supr-Infr 
    for fvalue,cvalue in feature_t,classes:
        idx = int((fvalue-Infr)/Blank*BinNum)
        QuantifiedSketchVector[idx,cvalue]=QuantifiedSketchVector[idx,cvalue]+1
    return QuantifiedSketchVector

def splitDataSet(DataSet,ratio):
    rows=DataSet['data'].shape[0]
    train_num=int(rows * ratio)
    TrainSet,TestSet={},{}
    TrainSet['data']=DataSet['data'][:train_num]
    TrainSet['target']=DataSet['target'][train_num:]
    TestSet['data']=DataSet['data'][train_num:]
    TestSet['target']=DataSet['target'][train_num:]
    return TrainSet,TestSet

def getTransformedSet(DataSet,features,feature_t):
    EstimatedSet=copy.deepcopy(DataSet)
    EstimatedSet['data']=np.delete(EstimatedSet['data'],features,axis=1)
    EstimatedSet['data']=np.insert(EstimatedSet['data'],0,values=feature_t,axis=1)
    return EstimatedSet



def generate_binary_training_samples(TransformationDataSets,Transformations,OriginalSet,BinNum,SampleNum,Improvement):
    # 分割数据集为训练集和测试集
    OriginalTrainSet,OriginalTestSet=splitDataSet(OriginalSet,0.8)
    # 计算基准准确率
    BenchClassifier=RandomForestClassifier(n_estimators=10)
    BenchClassifier.fit(OriginalTrainSet['data'],OriginalTrainSet['target'])
    BenchScore = BenchClassifier.score(OriginalTestSet['data'],OriginalTestSet['target'])
    # 两个标签,左边是有用,右边是无用
    UsefulTag,UselessTag = [1,0],[0,1]
    col_num = OriginalSet['data'].shape[1]
    ClassNum = len(np.unique(OriginalSet['target']))
    for sample in range(SampleNum):
        f1,f2 = random.sample([i for i in range(col_num)],2)
        feature_1,feature_2=OriginalSet['data'][:,f1],OriginalSet['data'][:,f2]
        #遍历每一种变换，为每一个二元MLP增加训练样本
        for trans_name,trans_verb in Transformations.binary_transformation_map.keys(),Transformations.binary_transformation_map.values():
            feature_t=trans_verb(feature_1,feature_2)
            QuantifiedSketchVector=getSketch(feature_t,BinNum,ClassNum,OriginalSet['target'])
            #产生转化后的数据集
            EstimatedSet=getTransformedSet(OriginalSet,[f1,f2],col_num)
            EstimatedTrainSet,EstimatedTestSet=splitDataSet(EstimatedSet,0.8)

            BenchClassifier.fit(EstimatedSet['data'],EstimatedSet['target'])
            EstimatedScore=BenchClassifier.score(EstimatedTestSet['data'],EstimatedTestSet['target'])
            TransformationDataSets[trans_name]['data'].append(QuantifiedSketchVector)
            if EstimatedScore-BenchScore > Improvement:
                TransformationDataSets[trans_name]['target'].append(UsefulTag)
            else:
                TransformationDataSets[trans_name]['target'].append(UselessTag)
    return TransformationDataSets

def generate_unary_training_samples(TransformationDataSets,Transformations,OriginalSet,BinNum,SampleNum,Improvement):
    #分割数据集为训练集和测试集
    OriginalTrainSet,OriginalTestSet=splitDataSet(OriginalSet,0.8)
    #计算基准准确率
    BenchClassifier = RandomForestClassifier(n_estimators=10)
    BenchClassifier.fit(OriginalTrainSet['data'],OriginalTrainSet['target'])
    BenchScore = BenchClassifier.score(OriginalTestSet['data'],OriginalTestSet['target'])
    #两个标签,左边是有用,右边是无用
    UsefulTag,UselessTag = [1,0],[0,1]
    col_num = OriginalSet['data'].shape[1]
    ClassNum = len(np.unique(OriginalSet['target']))
    for sample in range(SampleNum):
        f = random.sample([i for i in range(col_num)],1)
        feature=OriginalSet['data'][:,f]
        #遍历每一种变换,为每一个一元MLP增加样本
        for trans_name,trans_verb in Transformations.unary_transformation_map.keys(),Transformations.unary_transformation_map.values():
            feature_t=trans_verb(feature)
            QuantifiedSketchVector=getSketch(feature_t,BinNUm,ClassNum,OriginalSet['target'])
            #产生转化后的数据集
            EsitimatedSet=getTransformedSet(OriginalSet,[f],col_num)

            BenchClassifier.fit(EstimatedSet['data'],EstimatedSet['target'])
            EstimatedScore=BenchClassifier.score(EstimatedTestSet['data'],EstimatedTestSet['target'])
            if EstimatedScore-BenchScore > Improvement:
                TransformationDataSets[trans_name]['target'].append(UsefulTag)
            else:
                TransformationDataSets[trans_name]['target'].append(UselessTag)
    return TransformationDataSets




a=np.array([[1,2,3],[4,5,6],[7,8,9]])
a=np.delete(a,[0,1],axis=1)
print(a)


hyparams = {
    'bin_num':10,
    'improvement':0.01,
    'SampleNum':10000
}

feature_map=read_data_features('./utils/CleanedData.csv')
# 加载对应的transformation集合
Transformations=transformations()
Transformations.load_transformations()
#TrainSet = generate_training_samples(feature_map,hyparams['bin_num'])
