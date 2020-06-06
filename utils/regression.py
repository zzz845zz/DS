import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

def get_model_LinearRegression(x_train, y_train, normalize=False):
    model = LinearRegression(normalize=normalize)
    model.fit(X=x_train, y=y_train)
    return model

def get_model_ElasticNet(x_train, y_train, normalize=False):
    model = ElasticNet(normalize=normalize)
    model.fit(X=x_train, y=y_train)
    return model

def get_model_SVR(x_train, y_train):

    # regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    model = make_pipeline(SVR(C=1.0, epsilon=0.2))
    model.fit(x_train, y_train)
    return model


def get_model_MLPRegression(x_train, y_train, hidden_layers, max_iter, alpha, random_state):
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        max_iter = max_iter,
        alpha = alpha,
        random_state = random_state
    )
    model.fit(x_train, y_train)
    return model
    
    
    