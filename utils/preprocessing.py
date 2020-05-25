import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_specific_dong(df, dongname='중앙동'):
    is_matched = df['행정동']==dongname
    
    return df[is_matched]

def split_traintest(df, criteria='연월', threshold=202000):
    is_train = df[criteria] < threshold
    is_test = df[criteria] >= threshold
    
    train = df[is_train]
    test = df[is_test]
    
    return train, test

def split_xy(df, x_header=None, y_header=None, include_region=False):
    is_valid = df[y_header] != 0
    df = df[is_valid]
    
    x_header = [x for x in df.columns if x[0]=='남' or x[0]=='여' or x=='연월'] if x_header==None else x_header
    y_header = y_header if y_header==None else y_header
    
    x = np.array(df[x_header])
    y = np.array(df[y_header])
    return x, y


def normalize(data, method=''):
    
    data = data
    return data