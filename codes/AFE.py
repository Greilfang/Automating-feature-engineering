import numpy as np
from sklearn.neural_network import MLPClassifier
from utils.ReadRecords import read_data_features


def get_bin_index(Val,Supr,Infr,Blank):
    return int(Val-Infr)/Blank


def generate_samples(SubFeatureMap,BinNum,ClassNum):
    assert(type(SubFeatureMap)==dict)
    for FName,FValue in SubFeatureMap.keys(),SubFeatureMap.values():
        QuantifiedFValue=np.zeros((BinNum,ClassNum))
        Supr,Infr=max(FValue),min(FValue)
        for val in FValue:
            Blank=(Supr-Infr)/BinNum
            idx = get_bin_index(val,Infr,Supr,Blank)
            #cls = SubFeature
            QuantifiedFValue[idx]=QuantifiedFValue[idx]+1
        return QuantifiedFValue








hyparams = {
    'bucket_num':10
}

feature_map=read_data_features('./utils/CleanedData.csv')
print(feature_map)