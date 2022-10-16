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
from collections import defaultdict
import mlrose_hiive
import time
from random import randint
max_iters = 5000
rand_state = 2022

algos_dict = {'RHC': mlrose_hiive.random_hill_climb
                  ,'SA': mlrose_hiive.simulated_annealing
                  ,'GA': mlrose_hiive.genetic_alg
                  ,'MIMIC':  mlrose_hiive.mimic}

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

def solve_prob_all(problem):
    
    algos = algos_dict.keys()
    time_dict = dict.fromkeys(algos, 0)
    best_fitness_dict = dict.fromkeys(algos, 0)
    fitcurve_dict = dict.fromkeys(algos, [])
    for algo, algo_api in algos_dict.items(): 
        start_time = time.time()
        _, best_fitness, fitness_curve = algo_api(problem = problem
                                                 , max_attempts = 10
                                                 , curve = True
                                                 , max_iters = max_iters)
        end_time = time.time()
        best_fitness_dict[algo] = best_fitness
        time_dict[algo] = round(end_time - start_time,3)
        fitcurve_dict[algo] = fitness_curve
    return time_dict, best_fitness_dict, fitcurve_dict

def solve_prob(problem
              , algo
              , kwargs = {}):
    kwargs['problem'] = problem
    kwargs['curve'] = True
    kwargs['max_iters'] = max_iters
    start_time = time.time()
    _, best_fitness, fitness_curve = algo(**kwargs)
    end_time = time.time()
    return round(end_time - start_time,3), best_fitness, fitness_curve

def plot_func(d, title = 'Figure', xlabel = 'iteration', ylabel = 'fitness'):
    plt.figure(figsize = (10,8))
    for algo, y in d.items():
        plt.plot(y,label=algo)
    plt.ylabel(ylabel,size=14)
    plt.xlabel(xlabel,size=14)
    plt.legend()
    plt.title(title,size=14)
    plt.show()

def arr_add(x,y):
    if len(x)<=len(y):
        return np.append(x, [x[-1] for i in range(len(y) - len(x))]) + y
    else:
        return arr_add(y,x)
    
def para_tune(problem
              , algo_key
              , paras
              , trials = 10
              , kwargs={}
             ):
    para_name = paras[0]
    res_d = {}
    for v in paras[1]:
        
        curve = np.zeros(len(paras[1]))
        time = 0
        best_fit = 0
        for i in range(trials):
            kwargs[para_name] = v
            t, b, f = solve_prob(problem, algos_dict[algo_key], kwargs)
            time+=t
            best_fit+=b
            curve = arr_add(curve,f[:,0])
        print(f'{algo_key} with {para_name} = {v}: time: {time/trials}, best fitness: {best_fit/trials}')
        res_d[f'{para_name}: ' + str(v)]=curve/trials
    plot_func(res_d, title = f'{algo_key} with different {para_name}')