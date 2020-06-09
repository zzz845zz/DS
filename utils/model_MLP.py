import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')
import os

epo = 0
val_mse = 0
def build_model(input_shape, n_hidden=2, lr=0.00001):
    assert n_hidden in [2, 3], 'n_hidden error : %d' %(n_hidden)
    if n_hidden==2:
        model = keras.Sequential([
            layers.Dense(12, activation='relu', input_shape=input_shape),
            layers.Dense(6, activation='relu'),
            layers.Dense(1)
            ])
    elif n_hidden==3:
        model = keras.Sequential([
            layers.Dense(24, activation='relu', input_shape=input_shape),
            layers.Dense(12, activation='relu'),
            layers.Dense(6, activation='relu'),
            layers.Dense(1)
            ])

    optimizer = keras.optimizers.Adam(learning_rate=lr)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model

def train(model, model_name, dataset_name, x_train, y_train, x_test, y_test, epoch, batchsize):
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            global epo
            global val_mse
            epo = epoch
            val_mse = logs['val_mse']
            if epoch % 50 == 0:
                print('.', end='')
                #print(logs)

    # epoche 끝날때마다 모델 저장
    # ModelCheck = ModelCheckpoint(os.path.join('./log', 'MLP_normZ'+'-{epoch:04d}-{val_mse:.4f}.hdf5'), monitor='val_mse', verbose=0, 
    #                          save_best_only=True, save_weights_only=False, mode='auto', period=1)

    # metric의 상승이 멈출때마다 learning rate 낮춤
    ReduceLR = ReduceLROnPlateau(monitor='val_mse', factor=0.2, mode='auto',
                              patience=10, min_lr=1e-6, verbose=1)
    EarlyStop = EarlyStopping(monitor='val_mse', mode='auto', patience=50, restore_best_weights=True)

    EPOCHS = epoch
    history = model.fit(
        x_train, y_train, 
        batch_size = batchsize,
        epochs = epoch, 
        verbose = 0,
        validation_data = (x_test, y_test),
        callbacks=[PrintDot(), ReduceLR, EarlyStop])

    modelpath = os.path.join('./log', dataset_name, '%s-epoch:%04d-val_mse:%.4f.hdf5' %(model_name, epo, val_mse))
    model.save(modelpath)
    print(modelpath, 'saved')
    return model, history

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(14,6))

    plt.subplot(1,2,1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
    plt.ylim([0,np.max(hist['val_mae'])+2])
    plt.legend()

    plt.subplot(1,2,2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
    plt.ylim([0,np.max(hist['val_mse'])+2])
    plt.legend()
    plt.show()