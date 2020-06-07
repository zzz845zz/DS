import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_specific_dong(df, dongname='중앙동'):
    is_matched = df['행정동']==dongname
    
    return df[is_matched]

def split_train_val_test_by_file(dict_df={}, trainfiles=[], validatefiles=[], testfiles=[], category=None):
    train = pd.DataFrame()
    validate = pd.DataFrame()
    test = pd.DataFrame()

    for f in trainfiles:
        target = dict_df[f] if category==None else dict_df[f][dict_df[f]['서비스_업종_코드_명']==category]
        
        prev_shape = train.shape
        train = train.append(target)
        print('[train] %s : %s, accumulate : %s' %(f, target.shape, train.shape))
        assert (prev_shape[0] + target.shape[0]) == train.shape[0]
    
    for f in validatefiles:
        target = dict_df[f] if category==None else dict_df[f][dict_df[f]['서비스_업종_코드_명']==category]
        
        prev_shape = validate.shape
        validate = validate.append(target)
        print('[validate] %s : %s, accumulate : %s' %(f, target.shape, validate.shape))
        assert (prev_shape[0] + target.shape[0]) == validate.shape[0]
        
    for f in testfiles:
        target = dict_df[f] if category==None else dict_df[f][dict_df[f]['서비스_업종_코드_명']==category]
        
        prev_shape = test.shape
        test = test.append(target)
        print('[test] %s : %s, accumulate : %s' %(f, target.shape, test.shape))
        assert (prev_shape[0] + target.shape[0]) == test.shape[0]
        
    return train, validate, test

def split_traintest_by_column(df, criteria='연월', threshold=202000):
    is_train = df[criteria] < threshold
    is_test = df[criteria] >= threshold
    
    train = df[is_train]
    test = df[is_test]
    
    return train, test


def __get_next_quarter(year, quarter):
    quarter = (quarter+1) % 5
    
    if quarter==0:
        year += 1
        quarter = 1
    return year, quarter

def split_xy(df, x_header=None, y_header=None):
    # is_valid = df[y_header] != 0
    # df = df[is_valid]
    
    x = np.array(df[x_header])
    y = np.array(df[y_header])
    return x, y


def normalize(data, method=''):
    
    data = data
    return data