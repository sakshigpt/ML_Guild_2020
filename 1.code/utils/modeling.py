from sklearn import metrics
import numpy  as np
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.metrics import mean_squared_log_error


def print_result(pred_train, y_train, pred_test, y_test):
    rmse_train = np.sqrt(mean_squared_error(y_train, pred_train))
    r2_train = r2_score(y_train, pred_train)
    rmsle_train = np.sqrt(mean_squared_log_error(y_train, pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, pred_test))
    r2_test = r2_score(y_test, pred_test)
    rmsle_test = np.sqrt(mean_squared_log_error(y_test, pred_test))    
    print('Train RMSE: {}\nTrain R2: {}\nTrain RMSLE: {}\nTest RMSE: {}\nTest R2: {}\nTest RMSLE: {}'.
         format(rmse_train,r2_train,rmsle_train,rmse_test,r2_test,rmsle_test))
    return