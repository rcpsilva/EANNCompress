import random
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from scipy import stats

def rand(surrogate_ensemble,x,y):
    return random.choice(surrogate_ensemble)

def mse(surrogate_ensemble,x,y):
    return by_metric(surrogate_ensemble,mean_squared_error,x,y, metric_great_is_better=False)

def mape(surrogate_ensemble,x,y):
    return by_metric(surrogate_ensemble,mean_absolute_percentage_error,x,y, metric_great_is_better=False)

def r2(surrogate_ensemble,x,y):
    return by_metric(surrogate_ensemble,r2_score,x,y, metric_great_is_better=True)

def spearman(surrogate_ensemble,x,y):
    sp = lambda x, y : stats.spearmanr(x,y)[0]
    return by_metric(surrogate_ensemble,sp,x,y, metric_great_is_better=True)

def by_metric(surrogate_ensemble,metric,x,y, metric_great_is_better=True):
    """ Select the best surrogate using the input metric
    """
    if (type(metric) is not str):
      metric = make_scorer(metric, greater_is_better=metric_great_is_better)
    scores = np.zeros(len(surrogate_ensemble))
    for index,model in enumerate(surrogate_ensemble):
        model_scores = cross_val_score(model, x, y, scoring=metric)
        scores[index] = np.mean(model_scores)
    
    index_best_model = np.argmax(scores) if metric_great_is_better else np.argmin(scores)
    return surrogate_ensemble[index_best_model]



    
    

#### Example 
if __name__ == "__main__":

    surrogate_ensemble = [DecisionTreeRegressor(),
        LinearRegression(),
        KNeighborsRegressor()]

    x = np.random.rand(100,3) * 10
    y = x[:,0]**2 + x[:,1]**2 + x[:,2]**2

    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
    metric = mean_squared_error

    selected = by_metric(surrogate_ensemble,metric,X_train,y_train, metric_great_is_better=False)
    
    selected.fit(X_train, y_train)
    y_pred = selected.predict(X_test)
    accuracy = metric(y_test, y_pred)

    print(selected)
    print('Accuracy in the trainning set: {}'.format(accuracy))
