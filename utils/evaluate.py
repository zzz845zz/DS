import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# datas   : np.array
# headers : list of header name 
def visualize_CorrelCoeff_heatmap(datas, headers, figsize=(8, 8)):
    corrcoef = np.corrcoef(datas, rowvar=False)
        
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

def __get_error(y_gt, y_pred, title=''):
    li_error = []
    for i in zip(y_gt, y_pred):
        error = (abs(i[0]-i[1])/i[0])*100
        li_error.append(error)

    print(sum(li_error), y_pred.shape[0])
    print('평균 오차 %.lf%%' %(np.mean(li_error)))
    print('최대 오차 %.lf%%' %(np.max(li_error)))
    print('최소 오차 %.lf%%' %(np.min(li_error)))

    #li_error.sort(reverse=True)
    #print('Top 10 오차 %s' %(li_error[:10]))
    plt.plot(li_error)
    #plt.ylim([0, 100])
    plt.title(title)
    return li_error

def visualize_scatter(value_gt, value_predict, title='Regression'):
    
    if np.max(value_gt) > 10**6:
        value_gt = np.divide(value_gt, 10**6) 
        value_predict = np.divide(value_predict, 10**6) 
        title = '%s %s' %(title, '10^6')
    
    plt.scatter(x=value_gt, y=value_predict)
    tmp = np.arange(np.max(value_gt))
    plt.plot(tmp, tmp, "r-")
    plt.title('%s x=gt, y=predict' %(title))
    
    
def eval_regression(value_gt, value_predict, scaler=None, model_name=None):
    if scaler:
        value_gt = value_gt.reshape(value_gt.size, 1)
        value_predict = value_predict.reshape(value_predict.size, 1)
        value_gt = scaler.inverse_transform(value_gt)
        value_predict = scaler.inverse_transform(value_predict)
        
    RMSE = get_RMSE(value_gt, value_predict)
    R2 = get_R2(value_gt, value_predict)
    
    print('------- evaluate %s -------' %(model_name))
    print('RMSE : %f' %(RMSE))
    print('R2 : %f' %(R2))
    
    f, axs = plt.subplots(2,2,figsize=(14,6))
    plt.subplot(1, 2, 1)
    visualize_scatter(value_gt, value_predict, title=model_name)
    
    plt.subplot(1, 2, 2)
    print('----------------------------')
    return __get_error(value_gt, value_predict, title='%s error' %(model_name))
