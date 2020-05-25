import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# datas   : np.array
# headers : list of header name 
def visualize_CorrelCoeff_heatmap(datas, headers, figsize=(8, 8)):
    corrcoef = np.corrcoef(datas, rowvar=False)
#     for row in corrcoef:
#         for var in row:
#             print('%-10s ' %(var.round(2)), end='')
#         print()
        
    nvar = len(headers)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corrcoef)
    im.set_clim(-1, 1)
    ax.xaxis.set(ticks=np.arange(nvar), ticklabels=headers)
    ax.yaxis.set(ticks=np.arange(nvar), ticklabels=headers)
    cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')

    for row in range(nvar):
        for col in range(nvar):
            ax.text(col, row, corrcoef[row, col].round(2), ha='center', va='center')

    ax.grid(False)


# Root Mean Square Error
def get_RMSE(value_gt, value_predict):
    return np.sqrt(np.mean(np.square((value_gt.flatten()-value_predict.flatten()))))

def get_R2(value_gt, value_predict):
    return metrics.r2_score(value_gt.flatten(), value_predict.flatten())

def visualize_scatter(value_gt, value_predict, title='Regression'):
    plt.scatter(x=value_gt, y=value_predict)
    plt.title('%s x=gt, y=predict' %(title))
    
    
def eval_regression(value_gt, value_predict, model_name='None'):
    RMSE = get_RMSE(value_gt, value_predict)
    R2 = get_R2(value_gt, value_predict)
    
    print('------- evaluate %s -------' %(model_name))
    print('RMSE : %f' %(RMSE))
    print('R2 : %f' %(R2))
    
    visualize_scatter(value_gt, value_predict, title=model_name)
    print('----------------------------')