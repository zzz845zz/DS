import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings(action='ignore')

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow import keras
from tensorflow.keras import layers
import lightgbm as lgb

from utils import preprocessing as my_prep
from utils import model_Linear as my_Linear
from utils import model_MLP as my_MLP
from utils import evaluate as my_eval

# Config
flag_y_next_quarter = True
flag_random_split = True
PREDICT_CATEGORY = '편의점'   # None means overall

# Data load
def __get_data_dict(filepath):
    if flag_y_next_quarter:
        data_new = pd.read_csv(filepath)
    else:
        data_new = pd.read_csv(filepath, encoding='euc-kr')
    data_new.shape

    ### 분기별 분할
    quarters = ['2017_1', '2017_2', '2017_3', '2017_4', '2018_1', '2018_2', '2018_3', '2018_4', '2019_1', '2019_2', '2019_3']
    data_dict = {}
    for q in quarters:
        year, quarter = map(int, q.split("_"))

        bool_year = data_new.기준_년_코드==year
        bool_quarter = data_new.기준_분기_코드==quarter
        data_dict[q] = data_new[bool_year & bool_quarter] 
    return data_dict

# Split train, validate, test
def __get_train_test(data_dict):
    trainfiles = ['2017_1', '2017_2', '2017_3', '2017_4', '2018_1', '2018_2', '2018_3']
    validatefiles = []
    testfiles = ['2019_1', '2019_2', '2019_3']  # 2019_1, 2019_2, 2019_3, 2019_4 맞추기
    
    if flag_y_next_quarter:
        testfiles.append('2018_4')
    else:
        trainfiles.append('2018_4')
        testfiles.append('2019_4')

    train, validate, test = my_prep.split_train_val_test_by_file(data_dict, trainfiles, validatefiles, testfiles, category=PREDICT_CATEGORY)

    ### split x, y
    #x_header = [x for x in train.columns if '연령대' in x and x.find('연령대')==0]
    x_header = [x for x in train.columns if '남성연령대' in x or '여성연령대' in x]

    y_header = ['다음분기_매출_금액'] if flag_y_next_quarter else ['당월_매출_금액']
    print('x_header', x_header, 'y_header', y_header)

    x_train, y_train = my_prep.split_xy(train, x_header, y_header)
    x_test, y_test = my_prep.split_xy(test, x_header, y_header)

    ### Option(random split)
    if flag_random_split:
        print('random split')
        from sklearn.model_selection import train_test_split
        x_train = np.concatenate((x_train, x_test))
        y_train = np.concatenate((y_train, y_test))
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3)
        
    return x_train, y_train, x_test, y_test

# Normalize, PCA
def __get_normed(train, test, scaler):
    ### Normalize
    scaler = scaler.fit(train)
    train_normed = scaler.transform(train)
    test_normed = scaler.transform(test)
    
    return train_normed, test_normed

### PCA
def ___get_PCA(train, test, pca):
    pca.fit(train)
    train_pca = pca.transform(train)
    test_pca = pca.transform(test)

    print('pca ratios', pca.explained_variance_ratio_.round(2))

    top_n = 0
    ratio_accumul = 0
    for ix, ratio in enumerate(pca.explained_variance_ratio_.round(2)):
        ratio_accumul += ratio
        if ratio_accumul>0.9:
            top_n = ix+1
            break
    print('pca ratios', pca.explained_variance_ratio_.round(2))
    print('pca top-%d' %(top_n))
    train_pca_selected = train_pca[:, :top_n]
    test_pca_selected = test_pca[:, :top_n]
    return train_pca_selected, test_pca_selected


def get_traintest(filepath=None, pred_category=None, y_next_quarter=True, random_split=True, norm=None, pca=False):
    global PREDICT_CATEGORY
    global flag_y_next_quarter
    global flag_random_split
    flag_y_next_quarter = y_next_quarter
    flag_random_split = random_split
    PREDICT_CATEGORY = pred_category
    data_dict = __get_data_dict(filepath)
    x_train, y_train, x_test, y_test = __get_train_test(data_dict)
    
    # Normalize
    scaler = None
    if norm:
        assert norm in ['MinMax', 'Standard'], 'norm assign error : %s' %(norm)
        scaler = MinMaxScaler() if norm=='MinMax' else StandardScaler()
        x_train, x_test = __get_normed(x_train, x_test, scaler)
        y_train, y_test = __get_normed(y_train, y_test, scaler)
    
    # PCA
    if pca:
        assert norm=='Standard', 'pca True, but norm is not standard : %s' %(norm)
        x_train, x_test = ___get_PCA(x_train, x_test, PCA())
    
    return x_train, y_train, x_test, y_test, scaler

# datasets : [x_test, y_test, x_test, y_test]
def main(modelname=None, datasets=None, dataset_name=None, scaler=None, n_hidden=2, epoch=100, lr=0.0001):
    x_train, y_train, x_test, y_test = datasets
    
    # Train
    model_architectur = modelname.split("_")[0]
    assert modelname.split("_")[0] in ['LR', '4-MLP', '5-MLP', 'LGBM'], 'model assign error : %s' %(modelname)
    if model_architectur=='4-MLP':
        model = my_MLP.build_model(
            input_shape = [x_train.shape[1]],
            n_hidden=2,
            lr=lr
        )
        
        model, history = my_MLP.train(
            model, modelname, dataset_name,
            x_train, y_train, x_test, y_test,
            epoch=epoch, batchsize=32
        )
        
        #history = my_MLP.plot_history(history)
        return model, history
        
    elif model_architectur=='5-MLP':
        model = my_MLP.build_model(
            input_shape = [x_train.shape[1]],
            n_hidden=3,
            lr=lr
        )
        
        model, history = my_MLP.train(
            model, modelname, dataset_name,
            x_train, y_train, x_test, y_test,
            epoch=epoch, batchsize=32
        )
        
        #history = my_MLP.plot_history(history)
        return model, history
    
    elif model_architectur=='LGBM':
        model = lgb.LGBMRegressor()
        model.fit(x_train, y_train)
    
    # Evaluate
    #my_eval.eval_regression(y_test, model.predict(x_test), scaler=scaler, model_name=model)
    return model