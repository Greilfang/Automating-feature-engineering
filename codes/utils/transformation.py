# 这个文件是用来定义各种对feature 的变换

#储存每个MLP的数据集
TransformationDataSets={
    'sum':{'data':[],'target':[],'name':'sum'},
    'subtraction':{'data':[],'target':[],'name':'subtraction'},
    'multiplication':{'data':[],'target':[],'name':'multiplication'},
    'division':{'data':[],'target':[],'name':'devision'},
    'log':{'data':[],'target':[],'name':'log'},
    'frequency':{'data':[],'target':[],'name':'frequency'},
    'square':{'data':[],'target':[],'name':'square'},
    'round':{'data':[],'target':[],'name':'round'},
    'tanh':{'data':[],'target':[],'name':'tanh'},
    'sigmoid':{'data':[],'target':[],'name':'sigmoid'},
    'isotonic_regression':{'data':[],'target':[],'name':'isotomic_regression'},
    'zscore':{'data':[],'target':[],'name':'zscore'},
    'normalization':{'data':[],'target':[],'name':'normalization'},
}
class transformations:
    def __init__(self):
        self.binary_transformation_map={
            'sum':None,
            'subtraction':None,
            'multiplication':None,
            'division':None
        }
        self.unary_transformation_map={
            'log':None,
            'square_root':None,
            'frequency':None,
            'square':None,
            'round':None,
            'tanh':None,
            'sigmoid':None,
            'isotonic_regression':None,
            'zscore':None,
            'normalization':None
        }


    '''
    in: 一个ndarray的2列(m×1)
    out: 一个m×1 的 1 列
    description: 2列每个元素对应相加
    '''
    def sum(self,column_1,column_2):
        pass
    '''
    in: 一个ndarray的2列(m×1)
    out: 一个m×1 的 1 列
    description: 2列每个元素对应相减
    '''
    def substract(self,column_1,column_2):
        pass
    
    '''
    in: 一个ndarray的2列(m×1)
    out: 一个m×1 的 1 列
    description: 2列每个元素对应相减
    '''
    def multiply(self,column_1,column_2):
        pass

    '''
    in:  一个ndarray的2列(m×1)
    out: 一个m×1 的 1 列
    description: 2列每个元素对应相减
    '''
    def divide(self,column_1,column_2):
        pass
    '''
    in:  一个ndarray 的1列(m×1)
    out: 一个m×1 的 1 列
    description: 2列每个元素对应log_2^val
    '''
    def log(self,column):
        pass
    '''
    in:  一个ndarray 的1列(m×1)
    out: 一个m×1 的 1 列
    description: 2列每个元素对应求平方根,负值对绝对值求平方根加符号,例如square_root(-9)=-3
    '''
    def square_root(self,column):
        pass
    '''
    in:  一个ndarray 的1列(m×1)
    out: 一个m×1 的 1 列
    description: 对应元素替换成该元素在这一列出现的频次,例:[7,7,2,3,3,4] -> [2,2,1,2,2,1]
    '''
    def frequency(self,column):
        pass
    '''
    in:  一个ndarray 的1列(m×1)
    out: 一个m×1 的 1 列
    description:每个值对应四舍五入
    '''
    def round(self,column):
        pass
    '''
    in:  一个ndarray 的1列(m×1)
    out: 一个m×1 的 1 列
    description:每个值对应双曲正切
    '''
    def tanh(self,column):
        pass
    '''
    in:  一个ndarray 的1列(m×1)
    out: 一个m×1 的 1 列
    description:每个值对应sigmoid,自己查一下sigmoid函数
    '''
    def sigmoid(self,column):
        pass
    '''
    in:  一个ndarray 的1列(m×1),
    out: 一个m×1 的 1 列
    description:对该列值的分布进行,自己查一下保序回归
    '''
    def isotonic_regression(self,column):
        pass
    '''
    in:  一个ndarray 的1列(m×1),
    out: 一个m×1 的 1 列
    description:对该列值的分布进行z分数,查一下z分数
    '''    
    def zscore(self,column):
        pass
    '''
    in:  一个ndarray 的1列(m×1),
    out: 一个m×1 的 1 列
    description:对该列值的分布进行-1到1正则化,查一下normalization
    '''    
    def normalize(self,column):
        pass

    def load_transformations(self):
        self.unary_transformation_map['sum']=self.sum
        self.unary_transformation_map['substraction']=self.substract
        self.unary_transformation_map['multiplication']=self.multiply
        self.unary_transformation_map['division']=self.divide
        self.binary_transformation_map['log']=self.log
        self.binary_transformation_map['square']=self.square_root
        self.binary_transformation_map['frequency']=self.round
        self.binary_transformation_map['round']=self.round
        self.binary_transformation_map['tanh']=self.tanh
        self.binary_transformation_map['sigmoid']=self.sigmoid
        self.binary_transformation_map['isotonic_regression']=self.isotonic_regression
        self.binary_transformation_map['zscore']=self.zscore
        self.binary_transformation_map['normalization']=self.normalize
        print('load successfully')

#Transformations = transformations()
#Transformations.load_transformations()
