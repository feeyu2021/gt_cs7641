import pandas as pd
import numpy as np
import math

from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNNC
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier as MLP

from matplotlib import pyplot as plt
rand_state = 2022

def preprocess(data_raw, cat_cols = [], num_cols = [], y_col = 'y', drop_cols = []):
    data = data_raw.copy()
    
    for c in cat_cols:
        data.loc[data[c].isnull(),c] = data[c].mode().values[0]
    for c in num_cols:
        data.loc[data[c].isnull(),c] = data[c].median()
    data = pd.get_dummies(data, columns=cat_cols, drop_first = True)
    scaler = MinMaxScaler()
    data[num_cols] = scaler.fit_transform(data[num_cols])
        
    return data.drop(columns=drop_cols)

def train_test(data, estimator, params_grid, cv=5,metric = ['roc_auc','neg_log_loss']):
    clf = GridSearchCV(estimator
                   , params_grid
                   ,cv = cv
                   ,return_train_score = True
                   ,scoring = metric,refit =False)
    clf.fit(data.drop(columns=['y']), data['y'])
    
    for k,v in params_grid.items():
        if len(v)>1:
            para_name = k
            para_vals = v.copy()
            break
    for m in metric:
        print(f'''train {m} score: {clf.cv_results_['mean_train_'+m]}''')
        print(f'''test {m} score: {clf.cv_results_['mean_test_'+m]}''')
        plt.figure(figsize=(10,8))
        plt.plot(para_vals, clf.cv_results_['mean_train_'+m],'b-o',label='train')
        plt.plot(para_vals, clf.cv_results_['mean_test_'+m],'r:+',label='test')
        plt.ylabel('metric score',size=14)
        plt.xlabel(para_name,size=14)
        plt.legend()
        plt.title(m,size=14)
        plt.show()
    print(f'''mean fit time: {clf.cv_results_['mean_fit_time']}''')
    
    plt.figure(figsize=(10,8))
    plt.plot(para_vals, clf.cv_results_['mean_fit_time'],'b-o')
    plt.ylabel('mean_fit_time (s)',size=14)
    plt.xlabel(para_name,size=14)
    plt.title('mean_fit_time',size=14)