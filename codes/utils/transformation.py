# 这个文件是用来定义各种对feature 的变换
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
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
    #print(Supr,'si',Infr)
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
    def __init__(self):
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
        num=0
        # 分割数据集为训练集和测试集
        OriginalTrainSet, OriginalTestSet = splitDataSet(OriginalSet, 0.8)
        # 计算基准准确率
        BenchClassifier = RandomForestClassifier(n_estimators=10)
        BenchClassifier.fit(OriginalTrainSet['data'], OriginalTrainSet['target'])
        BenchScore = BenchClassifier.score(
            OriginalTestSet['data'], OriginalTestSet['target'])
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
                print('Trained Samples: ',sample)
                print('positive_num:',num)            
            for trans_name, trans_verb in self.unary_transformation_map.items():
                print('verb:',trans_verb)
                print('feature:',feature)
                feature_t = trans_verb(feature)
                if len(feature_t) == 0:
                    continue
                QuantifiedSketchVector = getSketch(
                    feature_t, BinNum, ClassNum, OriginalSet['target'])
                # 产生转化后的数据集
                EstimatedSet = getTransformedSet(OriginalSet, [f], col_num)
                EstimatedTrainSet, EstimatedTestSet = splitDataSet(
                    EstimatedSet, 0.8)
                EstimatedClassifier = RandomForestClassifier(n_estimators=10)
                EstimatedClassifier.fit(
                    EstimatedTrainSet['data'], EstimatedTrainSet['target'])
                EstimatedScore = EstimatedClassifier.score(
                    EstimatedTestSet['data'], EstimatedTestSet['target'])
                self.DataSets[trans_name]['data'].append(
                    QuantifiedSketchVector)
                if EstimatedScore-BenchScore > Improvement:
                    self.DataSets[trans_name]['target'].append(UsefulTag)
                    print('trans_name:',trans_name[:3],'useful:',sample,'score',EstimatedScore)
                    num=num+1
                else:
                    self.DataSets[trans_name]['target'].append(UselessTag)
                    print('trans_name:',trans_name[:3],'useful:',sample,'score',EstimatedScore)
    
    
    def generate_binary_training_samples(self, OriginalSet, BinNum, SampleNum, Improvement):
        num=0
        # 分割数据集为训练集和测试集
        OriginalTrainSet, OriginalTestSet = splitDataSet(OriginalSet, 0.8)
        # 计算基准准确率
        BenchClassifier = RandomForestClassifier(n_estimators=10)
        BenchClassifier.fit(OriginalTrainSet['data'], OriginalTrainSet['target'])
        BenchScore = BenchClassifier.score(OriginalTestSet['data'], OriginalTestSet['target'])
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
                print('Trained Samples:',sample)
                print('positive_num:',num)
                            
            for trans_name, trans_verb in self.binary_transformation_map.items():
                feature_t = trans_verb(feature_1, feature_2)
                if len(feature_t)==0:
                    continue
                QuantifiedSketchVector = getSketch(
                    feature_t, BinNum, ClassNum, OriginalSet['target'])
                # 产生转化后的数据集
                EstimatedSet = getTransformedSet(OriginalSet, [f1, f2], col_num)
                EstimatedTrainSet, EstimatedTestSet = splitDataSet(
                    EstimatedSet, 0.8)

                EstimatedClassifier = RandomForestClassifier(n_estimators=10)
                EstimatedClassifier.fit(
                    EstimatedTrainSet['data'], EstimatedTrainSet['target'])
                EstimatedScore = EstimatedClassifier.score(
                    EstimatedTestSet['data'], EstimatedTestSet['target'])
                self.DataSets[trans_name]['data'].append(
                    QuantifiedSketchVector)
                if EstimatedScore-BenchScore > Improvement:
                    self.DataSets[trans_name]['target'].append(UsefulTag)
                    print('trans_name:',trans_name[:3],'√',sample,'score',EstimatedScore)
                    num=num+1
                else:
                    self.DataSets[trans_name]['target'].append(UselessTag)
                    print('trans_name:',trans_name[:3],'×',sample,'score',EstimatedScore)

    
    def generate_training_samples(self, OriginalSet, hyparams):
        BinNum, SampleNum, Improvement = hyparams['bin_num'], hyparams['sample_num'], hyparams['improvement']
        self.generate_unary_training_samples(OriginalSet, BinNum, SampleNum, Improvement)
        self.generate_binary_training_samples(OriginalSet, BinNum, SampleNum, Improvement)

        #对数据集内数据进行重新设置维数使其输入mlp
        for trans_name,trans_set in self.DataSets.items():
            trans_set['data']=list(map(lambda x: x.flatten(),trans_set['data']))
            trans_set['data']=np.array(trans_set['data'])
        
        trans_set['target']=np.array(trans_set('target'))

    


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


