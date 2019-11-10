# 这个文件是用来定义各种对feature 的变换
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random


def getSketch(feature_t, BinNum, cls_num, classes):
    QuantifiedSketchVector = np.zeros((BinNum, cls_num))
    Supr, Infr = max(feature_t), min(feature_t)
    Blank = Supr-Infr
    for fvalue, cvalue in feature_t, classes:
        idx = int((fvalue-Infr)/Blank*BinNum)
        QuantifiedSketchVector[idx,
                               cvalue] = QuantifiedSketchVector[idx, cvalue]+1
    return QuantifiedSketchVector


def splitDataSet(DataSet, ratio):
    rows = DataSet['data'].shape[0]
    train_num = int(rows * ratio)
    TrainSet, TestSet = {}, {}
    TrainSet['data'] = DataSet['data'][:train_num]
    TrainSet['target'] = DataSet['target'][train_num:]
    TestSet['data'] = DataSet['data'][train_num:]
    TestSet['target'] = DataSet['target'][train_num:]
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
            'subtraction': {'data': [], 'target': [], 'name': 'subtraction'},
            'multiplication': {'data': [], 'target': [], 'name': 'multiplication'},
            'division': {'data': [], 'target': [], 'name': 'devision'},
            'log': {'data': [], 'target': [], 'name': 'log'},
            'frequency': {'data': [], 'target': [], 'name': 'frequency'},
            'square': {'data': [], 'target': [], 'name': 'square'},
            'round': {'data': [], 'target': [], 'name': 'round'},
            'tanh': {'data': [], 'target': [], 'name': 'tanh'},
            'sigmoid': {'data': [], 'target': [], 'name': 'sigmoid'},
            'isotonic_regression': {'data': [], 'target': [], 'name': 'isotomic_regression'},
            'zscore': {'data': [], 'target': [], 'name': 'zscore'},
            'normalization': {'data': [], 'target': [], 'name': 'normalization'},

        }

        self.binary_transformation_map = {
            'sum': None,
            'subtraction': None,
            'multiplication': None,
            'division': None
        }
        self.unary_transformation_map = {
            'log': None,
            'square_root': None,
            'frequency': None,
            'square': None,
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
                hidden_layer_sizes=(400),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=5000, shuffle=True,
                random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        for tran in self.unary_transformation_map.keys():
            self.unary_MLPs[tran] = MLPClassifier(
                hidden_layer_sizes=(400),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=5000, shuffle=True,
                random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    def generate_unary_training_samples(self, OriginalSet, BinNum, SampleNum, Improvement):
        # 分割数据集为训练集和测试集
        OriginalTrainSet, OriginalTestSet = splitDataSet(OriginalSet, 0.8)
        # 计算基准准确率
        BenchClassifier = RandomForestClassifier(n_estimators=10)
        BenchClassifier.fit(OriginalTrainSet['data'], OriginalTrainSet['target'])
        BenchScore = BenchClassifier.score(
            OriginalTestSet['data'], OriginalTestSet['target'])
        # 两个标签,左边是有用,右边是无用
        UsefulTag, UselessTag = [1, 0], [0, 1]
        col_num = OriginalSet['data'].shape[1]
        ClassNum = len(np.unique(OriginalSet['target']))
        for sample in range(SampleNum):
            f = random.sample([i for i in range(col_num)], 1)
            feature = OriginalSet['data'][:, f]
            # 遍历每一种变换,为每一个一元MLP增加样本
            for trans_name, trans_verb in self.unary_transformation_map.keys(), self.unary_transformation_map.values():
                feature_t = trans_verb(feature)
                QuantifiedSketchVector = getSketch(
                    feature_t, BinNum, ClassNum, OriginalSet['target'])
                # 产生转化后的数据集
                EstimatedSet = getTransformedSet(OriginalSet, [f], col_num)
                EstimatedTrainSet, EstimatedTestSet = splitDataSet(
                    EstimatedSet, 0.8)
                BenchClassifier.fit(
                    EstimatedTrainSet['data'], EstimatedTrainSet['target'])
                EstimatedScore = BenchClassifier.score(
                    EstimatedTestSet['data'], EstimatedTestSet['target'])
                self.DataSets[trans_name]['data'].append(
                    QuantifiedSketchVector)
                if EstimatedScore-BenchScore > Improvement:
                    self.DataSets[trans_name]['target'].append(
                        UsefulTag)
                else:
                    self.DataSets[trans_name]['target'].append(
                        UselessTag)

    def generate_binary_training_samples(self, OriginalSet, BinNum, SampleNum, Improvement):
        # 分割数据集为训练集和测试集
        OriginalTrainSet, OriginalTestSet = splitDataSet(OriginalSet, 0.8)
        # 计算基准准确率
        BenchClassifier = RandomForestClassifier(n_estimators=10)
        BenchClassifier.fit(OriginalTrainSet['data'], OriginalTrainSet['target'])
        BenchScore = BenchClassifier.score(
            OriginalTestSet['data'], OriginalTestSet['target'])
        # 两个标签,左边是有用,右边是无用
        UsefulTag, UselessTag = [1, 0], [0, 1]
        col_num = OriginalSet['data'].shape[1]
        ClassNum = len(np.unique(OriginalSet['target']))
        for sample in range(SampleNum):
            f1, f2 = random.sample([i for i in range(col_num)], 2)
            feature_1, feature_2 = OriginalSet['data'][:,
                                                    f1], OriginalSet['data'][:, f2]
            # 遍历每一种变换，为每一个二元MLP增加训练样本
            for trans_name, trans_verb in self.binary_transformation_map.keys(), self.binary_transformation_map.values():
                feature_t = trans_verb(feature_1, feature_2)
                QuantifiedSketchVector = getSketch(
                    feature_t, BinNum, ClassNum, OriginalSet['target'])
                # 产生转化后的数据集
                EstimatedSet = getTransformedSet(OriginalSet, [f1, f2], col_num)
                EstimatedTrainSet, EstimatedTestSet = splitDataSet(
                    EstimatedSet, 0.8)

                BenchClassifier.fit(
                    EstimatedTrainSet['data'], EstimatedTrainSet['target'])
                EstimatedScore = BenchClassifier.score(
                    EstimatedTestSet['data'], EstimatedTestSet['target'])
                self.DataSets[trans_name]['data'].append(
                    QuantifiedSketchVector)
                if EstimatedScore-BenchScore > Improvement:
                    self.DataSets[trans_name]['target'].append(UsefulTag)
                else:
                    self.DataSets[trans_name]['target'].append(UselessTag)
    
    def generate_training_samples(self, OriginalSet, hyparams):
        BinNum, SampleNum, Improvement = hyparams['bin_num'], hyparams['sample_num'], hyparams['improvement']
        self.generate_binary_training_samples(OriginalSet, BinNum, SampleNum, Improvement)
        self.generate_unary_training_samples(OriginalSet, BinNum, SampleNum, Improvement)

    





    '''
    in: 一个ndarray的2列(m×1)
    out: 一个m×1 的 1 列
    description: 2列每个元素对应相加
    '''
    def sum(self, column_1, column_2):
        pass
    '''
    in: 一个ndarray的2列(m×1)
    out: 一个m×1 的 1 列
    description: 2列每个元素对应相减
    '''

    def substract(self, column_1, column_2):
        pass

    '''
    in: 一个ndarray的2列(m×1)
    out: 一个m×1 的 1 列
    description: 2列每个元素对应相减
    '''

    def multiply(self, column_1, column_2):
        pass

    '''
    in:  一个ndarray的2列(m×1)
    out: 一个m×1 的 1 列
    description: 2列每个元素对应相减
    '''

    def divide(self, column_1, column_2):
        pass
    '''
    in:  一个ndarray 的1列(m×1)
    out: 一个m×1 的 1 列
    description: 2列每个元素对应log_2^val
    '''

    def log(self, column):
        pass
    '''
    in:  一个ndarray 的1列(m×1)
    out: 一个m×1 的 1 列
    description: 2列每个元素对应求平方根,负值对绝对值求平方根加符号,例如square_root(-9)=-3
    '''

    def square_root(self, column):
        pass
    '''
    in:  一个ndarray 的1列(m×1)
    out: 一个m×1 的 1 列
    description: 对应元素替换成该元素在这一列出现的频次,例:[7,7,2,3,3,4] -> [2,2,1,2,2,1]
    '''

    def frequency(self, column):
        pass
    '''
    in:  一个ndarray 的1列(m×1)
    out: 一个m×1 的 1 列
    description:每个值对应四舍五入
    '''

    def round(self, column):
        pass
    '''
    in:  一个ndarray 的1列(m×1)
    out: 一个m×1 的 1 列
    description:每个值对应双曲正切
    '''

    def tanh(self, column):
        pass
    '''
    in:  一个ndarray 的1列(m×1)
    out: 一个m×1 的 1 列
    description:每个值对应sigmoid,自己查一下sigmoid函数
    '''

    def sigmoid(self, column):
        pass
    '''
    in:  一个ndarray 的1列(m×1),
    out: 一个m×1 的 1 列
    description:对该列值的分布进行,自己查一下保序回归
    '''

    def isotonic_regression(self, column):
        pass
    '''
    in:  一个ndarray 的1列(m×1),
    out: 一个m×1 的 1 列
    description:对该列值的分布进行z分数,查一下z分数
    '''

    def zscore(self, column):
        pass
    '''
    in:  一个ndarray 的1列(m×1),
    out: 一个m×1 的 1 列
    description:对该列值的分布进行-1到1正则化,查一下normalization
    '''

    def normalize(self, column):
        pass

    def upload(self):
        self.unary_transformation_map['sum'] = self.sum
        self.unary_transformation_map['substraction'] = self.substract
        self.unary_transformation_map['multiplication'] = self.multiply
        self.unary_transformation_map['division'] = self.divide
        self.binary_transformation_map['log'] = self.log
        self.binary_transformation_map['square'] = self.square_root
        self.binary_transformation_map['frequency'] = self.round
        self.binary_transformation_map['round'] = self.round
        self.binary_transformation_map['tanh'] = self.tanh
        self.binary_transformation_map['sigmoid'] = self.sigmoid
        self.binary_transformation_map['isotonic_regression'] = self.isotonic_regression
        self.binary_transformation_map['zscore'] = self.zscore
        self.binary_transformation_map['normalization'] = self.normalize
        print('load successfully')

#Transformations = transformations()
# Transformations.load_transformations()
