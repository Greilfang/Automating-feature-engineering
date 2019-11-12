# 这个文件是用来定义各种对feature 的变换
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from utils.ReadRecords import load_transformations,save_transformations
import numpy as np
import pandas as pd
import random
import copy
from sklearn.isotonic import IsotonicRegression


def getSketch(feature_t, BinNum, cls_num, classes):
    QuantifiedSketchVector = np.zeros((cls_num,BinNum+1))
    Supr, Infr = max(feature_t), min(feature_t)
    Blank = Supr-Infr
    if Blank == 0:return None
    for fvalue, cvalue in zip(feature_t, classes):
        #print(fvalue,'fv',cvalue)
        idx = int((fvalue-Infr)/Blank*BinNum)
        QuantifiedSketchVector[cvalue,idx] = QuantifiedSketchVector[cvalue,idx]+1
    QuantifiedSketchVector = np.delete(QuantifiedSketchVector, -1, axis=1)
    return QuantifiedSketchVector


def splitDataSet(DataSet, ratio):
    rows = DataSet['data'].shape[0]
    train_num = int(rows * ratio)
    TrainSet, TestSet = {}, {}
    TrainSet['data'] = DataSet['data'][:train_num]
    TrainSet['target'] = DataSet['target'][:train_num].astype('int')
    TestSet['data'] = DataSet['data'][train_num:]
    TestSet['target'] = DataSet['target'][train_num:].astype('int')
    return TrainSet, TestSet


def getTransformedSet(DataSet, features, feature_t):
    EstimatedSet = copy.deepcopy(DataSet)
    EstimatedSet['data'] = np.delete(EstimatedSet['data'], features, axis=1)
    EstimatedSet['data'] = np.insert(
        EstimatedSet['data'], 0, values=feature_t, axis=1)
    return EstimatedSet

# 储存每个MLP的数据集
class transformations:
    def __init__(self,hyparams):
        self.BinNum = hyparams['bin_num']
        self.DataSets = {
            'sum': {'data': [], 'target': [], 'name': 'sum'},
            'substraction': {'data': [], 'target': [], 'name': 'substraction'},
            'multiplication': {'data': [], 'target': [], 'name': 'multiplication'},
            'division': {'data': [], 'target': [], 'name': 'devision'},
            'log': {'data': [], 'target': [], 'name': 'log'},
            'frequency': {'data': [], 'target': [], 'name': 'frequency'},
            'square': {'data': [], 'target': [], 'name': 'square'},
            'square_root': {'data': [], 'target': [], 'name': 'square_root'},
            'round': {'data': [], 'target': [], 'name': 'round'},
            'tanh': {'data': [], 'target': [], 'name': 'tanh'},
            'sigmoid': {'data': [], 'target': [], 'name': 'sigmoid'},
            'isotonic_regression': {'data': [], 'target': [], 'name': 'isotomic_regression'},
            'zscore': {'data': [], 'target': [], 'name': 'zscore'},
            'normalization': {'data': [], 'target': [], 'name': 'normalization'},

        }

        self.binary_transformation_map = {
            'sum': None,
            'substraction': None,
            'multiplication': None,
            'division': None
        }
        self.unary_transformation_map = {
            'log': None,
            'square_root': None,
            'square':None,
            'frequency': None,
            'round': None,
            'tanh': None,
            'sigmoid': None,
            'isotonic_regression': None,
            'zscore': None,
            'normalization': None
        }
        self.binary_MLPs = {}
        self.unary_MLPs = {}
        # 初始化MLP感知器用于训练
        for tran in self.binary_transformation_map.keys():
            self.binary_MLPs[tran] = MLPClassifier(
                hidden_layer_sizes=(256),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=5000, shuffle=True,
                random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        for tran in self.unary_transformation_map.keys():
            self.unary_MLPs[tran] = MLPClassifier(
                hidden_layer_sizes=(256),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=5000, shuffle=True,
                random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    def generate_unary_training_samples(self, OriginalSet, BinNum, SampleNum, Improvement):
        negative_num,positive_num=0,0
        # 分割数据集为训练集和测试集
        #OriginalTrainSet, OriginalTestSet = splitDataSet(OriginalSet, 0.8)
        # 计算基准准确率
        BenchClassifier = RandomForestClassifier(n_estimators=10)
        BenchScore = cross_val_score(BenchClassifier, OriginalSet['data'], OriginalSet['target'], cv=5,scoring='f1').mean()
        print('unary_benchscore:',BenchScore)
        # 两个标签,左边是有用,右边是无用
        UsefulTag, UselessTag = np.array([1, 0]),np.array([0, 1])
        col_num = OriginalSet['data'].shape[1]
        ClassNum = len(np.unique(OriginalSet['target']))
        for sample in range(SampleNum):
            f = random.sample([i for i in range(col_num)], 1)[0]
            feature = OriginalSet['data'][:, f]
            # 遍历每一种变换,为每一个一元MLP增加样本
            if sample % 10 ==0:
                print('-------------------------------------')
                print('Trained Samples: ',sample)
                print('unary_positive_num:',positive_num)
                print('unary_negative_num:',negative_num)
                print('-------------------------------------')                
            for trans_name, trans_verb in self.unary_transformation_map.items():
                #print('verb:',trans_verb)
                #print('feature:',feature)
                feature_t = trans_verb(feature)
                if len(feature_t) == 0:
                    continue
                QuantifiedSketchVector = getSketch(
                    feature_t, BinNum, ClassNum, OriginalSet['target'])
                # 产生转化后的数据集
                if QuantifiedSketchVector is None:
                    continue
                
                EstimatedSet = getTransformedSet(OriginalSet, [f], col_num)
                EstimatedClassifier = RandomForestClassifier(n_estimators=10)
                EstimatedScore = cross_val_score(EstimatedClassifier, EstimatedSet['data'], EstimatedSet['target'], cv=5,scoring='f1').mean()


                self.DataSets[trans_name]['data'].append(QuantifiedSketchVector)
                if EstimatedScore-BenchScore > Improvement:
                    self.DataSets[trans_name]['target'].append(UsefulTag)
                    print('trans_name:',trans_name[:3],'Y:',sample,'score',EstimatedScore)
                    positive_num=positive_num+1
                else:
                    self.DataSets[trans_name]['target'].append(UselessTag)
                    print('trans_name:',trans_name[:3],'N:',sample,'score',EstimatedScore)
                    negative_num=negative_num+1
    
    
    def generate_binary_training_samples(self, OriginalSet, BinNum, SampleNum, Improvement):
        negative_num,positive_num=0,0
        # 计算基准准确率
        BenchClassifier = RandomForestClassifier(n_estimators=10)
        BenchScore = cross_val_score(BenchClassifier, OriginalSet['data'], OriginalSet['target'], cv=5,scoring='f1').mean()
        print('binary_benchscore',BenchScore)
        # 两个标签,左边是有用,右边是无用
        UsefulTag, UselessTag = np.array([1, 0]), np.array([0, 1])
        col_num = OriginalSet['data'].shape[1]
        ClassNum = len(np.unique(OriginalSet['target']))
        for sample in range(SampleNum):
            f1, f2 = random.sample([i for i in range(col_num)], 2)
            feature_1, feature_2 = OriginalSet['data'][:,f1], OriginalSet['data'][:, f2]
            # 遍历每一种变换，为每一个二元MLP增加训练样本
            if sample % 10 ==0:
                print('-------------------------------------')
                print('Trained Samples: ',sample)
                print('binary_positive_num:',positive_num)
                print('binary_negative_num:',negative_num)
                print('-------------------------------------')      
                            
            for trans_name, trans_verb in self.binary_transformation_map.items():
                #print('verb:',trans_verb)
                #print('feature:',feature)                
                feature_t = trans_verb(feature_1, feature_2)
                if len(feature_t)==0:
                    continue
                QuantifiedSketchVector = getSketch(
                    feature_t, BinNum, ClassNum, OriginalSet['target'])
                if QuantifiedSketchVector is None:
                    continue
                # 产生转化后的数据集
                EstimatedSet = getTransformedSet(OriginalSet, [f1, f2], col_num)
                EstimatedClassifier = RandomForestClassifier(n_estimators=10)
                EstimatedScore = cross_val_score(EstimatedClassifier, EstimatedSet['data'], EstimatedSet['target'], cv=5,scoring='f1').mean()
                
                self.DataSets[trans_name]['data'].append(QuantifiedSketchVector)
                if EstimatedScore-BenchScore > Improvement:
                    self.DataSets[trans_name]['target'].append(UsefulTag)
                    print('trans_name:',trans_name[:3],'Y:',sample,'score',EstimatedScore)
                    positive_num=positive_num+1
                else:
                    self.DataSets[trans_name]['target'].append(UselessTag)
                    print('trans_name:',trans_name[:3],'N:',sample,'score',EstimatedScore)
                    negative_num=negative_num+1

    
    def generate_training_samples(self, OriginalSets, hyparams):
        BinNum, UNum,BNum, Improvement = hyparams['bin_num'], hyparams['unary_sample_num'],hyparams['binary_sample_num'],hyparams['improvement']
        for OriginalSet in OriginalSets:
            print('--------------------------------------')
            print('DataSet Name:',OriginalSet['name'])
            self.generate_unary_training_samples(OriginalSet, BinNum, UNum, Improvement)
            self.generate_binary_training_samples(OriginalSet, BinNum, BNum, Improvement)

        #对数据集内数据进行重新设置维数使其输入mlp
        for trans_name,trans_set in self.DataSets.items():
            trans_set['data']=list(map(lambda x: x.flatten(),trans_set['data']))
            trans_set['data']=np.array(trans_set['data'])
            trans_set['target']=np.array(trans_set['target'])
    
    def predict(self,TargetSet,attempts,threshold):
        probs=[]
        col_num = TargetSet['data'].shape[1]
        ClassNum = len(np.unique(TargetSet['target']))
        for attempt in range(attempts):
            f1, f2 = random.sample([i for i in range(col_num)], 2)
            feature_1,feature_2=TargetSet['data'][:,f1],TargetSet['data'][:,f2]
            for name,MLP in self.binary_MLPs.items():
                feature_t=self.binary_transformation_map[name](feature_1,feature_2)
                QuantifiedSketchVector=getSketch(feature_t,self.BinNum,ClassNum,TargetSet['target'])
                if QuantifiedSketchVector is None:
                    continue
                prob = MLP.predic_proba(QuantifiedSketchVector)
                probs.append(prob)
            # 二元变化结束,开始一元变化
            f = random.sample([i for i in range(col_num)],1)[0]
            feature = TargetSet['data'][:,f]            
            for name,MLP in self.unary_MLPs.items():
                QuantifiedSketchVector=getSketch(feature,self.BinNum,ClassNum,TargetSet['target'])
                if QuantifiedSketchVector is None:
                    continue
                prob = MLP.predic_proba(QuantifiedSketchVector)
                probs.append(prob)
            # 输出概率
            print(probs)



    


    '''
    in: 一个ndarray的2列(m×1)
    out: 一个m×1 的 1 列
    description: 2列每个元素对应相加
    '''
    def sum(self, column_1, column_2):
        return column_1 + column_2
    '''
    in: 一个ndarray的2列(m×1)
    out: 一个m×1 的 1 列
    description: 2列每个元素对应相减
    '''

    def substract(self, column_1, column_2):
        return column_1-column_2

    '''
    in: 一个ndarray的2列(m×1)
    out: 一个m×1 的 1 列
    description: 2列每个元素对应相减
    '''

    def multiply(self, column_1, column_2):
        return column_1 * column_2

    '''
    in:  一个ndarray的2列(m×1)
    out: 一个m×1 的 1 列
    description: 2列每个元素对应相减
    '''

    def divide(self, column_1, column_2):
        if np.all(column_2!=0):
            return column_1/column_2
        return []
    '''
    in:  一个ndarray 的1列(m×1)
    out: 一个m×1 的 1 列
    description: 2列每个元素对应log_2^val
    '''

    def log(self, column):
        if np.all(column>0):
            return np.log2(column)
        return []
    '''
    in:  一个ndarray 的1列(m×1)
    out: 一个m×1 的 1 列
    description: 2列每个元素绝对值对应求平方根
    '''
    def square_root(self, column):
        return np.sqrt(np.abs(column))
    '''
    in:  一个ndarray 的1列(m×1)
    out: 一个m×1 的 1 列
    description: 2列每个元素对应求平方根,负值对绝对值求平方根加符号,例如square_root(-9)=-3
    '''
    def square(self, column):
        return np.sqrt(np.abs(column)) * np.sign(column)
    '''
    in:  一个ndarray 的1列(m×1)
    out: 一个m×1 的 1 列
    description: 对应元素替换成该元素在这一列出现的频次,例:[7,7,2,3,3,4] -> [2,2,1,2,2,1]
    '''

    def frequency(self, column):
        freq=pd.value_counts(column)
        return np.array(list(map(lambda x: freq[x],column)))
    '''
    in:  一个ndarray 的1列(m×1)
    out: 一个m×1 的 1 列
    description:每个值对应四舍五入
    '''
    def round(self, column):
        return np.round(column).astype('int')
    '''
    in:  一个ndarray 的1列(m×1)
    out: 一个m×1 的 1 列
    description:每个值对应双曲正切
    '''

    def tanh(self, column):
        return np.tanh(column)
    '''
    in:  一个ndarray 的1列(m×1)
    out: 一个m×1 的 1 列
    description:每个值对应sigmoid,自己查一下sigmoid函数
    '''

    def sigmoid(self, column):
        return(1/(1+np.exp(-column)))
    '''
    in:  一个ndarray 的1列(m×1),
    out: 一个m×1 的 1 列
    description:对该列值的分布进行,自己查一下保序回归
    '''

    def isotonic_regression(self, column):
        inds=range(column.shape[0])
        return IsotonicRegression().fit_transform(inds,column)
    '''
    in:  一个ndarray 的1列(m×1),
    out: 一个m×1 的 1 列
    description:对该列值的分布进行z分数,查一下z分数
    '''

    def zscore(self, column):
        mv=np.mean(column)
        stv = np.std(column)
        if stv!=0:
            return (column-mv)/stv
        return []
    '''
    in:  一个ndarray 的1列(m×1),
    out: 一个m×1 的 1 列
    description:对该列值的分布进行-1到1正则化,查一下normalization
    '''

    def normalize(self, column):
        maxv,minv=np.max(column),np.min(column)
        return -1 + 2/(maxv-minv) * (column-minv)

    def upload(self):
        self.binary_transformation_map['sum'] = self.sum
        self.binary_transformation_map['substraction'] = self.substract
        self.binary_transformation_map['multiplication'] = self.multiply
        self.binary_transformation_map['division'] = self.divide


        self.unary_transformation_map['log'] = self.log
        self.unary_transformation_map['square_root'] = self.square_root
        self.unary_transformation_map['square']=self.square
        self.unary_transformation_map['frequency'] = self.round
        self.unary_transformation_map['round'] = self.round
        self.unary_transformation_map['tanh'] = self.tanh
        self.unary_transformation_map['sigmoid'] = self.sigmoid
        self.unary_transformation_map['isotonic_regression'] = self.isotonic_regression
        self.unary_transformation_map['zscore'] = self.zscore
        self.unary_transformation_map['normalization'] = self.normalize
        print('load successfully')


