from sklearn import metrics
import numpy  as np
from sklearn.metrics import r2_score,mean_squared_error

def print_result(pred_train, y_train, pred_test, y_test):
    rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
    r2_train = r2_score(y_train, pred_train)
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
    r2_test = r2_score(y_test, pred_test)
    print('Train RMSE: {}\nTrain R2: {}\nTest RMSE: {}\nTest R2: {}'.
         format(rmse_train,r2_train,rmse_test,r2_test))
    return