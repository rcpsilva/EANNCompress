import random
import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

def rand(surrogate_ensemble,x,y):
    return random.choice(surrogate_ensemble)

def by_metric(surrogate_ensemble,metric,x,y, metric_great_is_better=True):
    """ Select the best surrogate using the input metric
    """
    if (type(metric) is not str):
      metric = make_scorer(metric, greater_is_better=metric_great_is_better)
    scores = np.zeros(len(surrogate_ensemble))
    for index,model in enumerate(surrogate_ensemble):
        model_scores = cross_val_score(model, x, y, scoring=metric)
        scores[index] = np.mean(model_scores)
    
    index_best_model = np.argmax(scores)
    return surrogate_ensemble[index_best_model]



    
    

#### Example 
if __name__ == "__main__":

    surrogate_ensemble = [DecisionTreeRegressor(),
        LinearRegression(),
        KNeighborsRegressor()]

    x = np.random.rand(100,3) * 10
    y = x[:,0]**2 + x[:,1]**2 + x[:,2]**2

    metric = mean_squared_error

    selected = by_metric(surrogate_ensemble,metric,x,y, metric_great_is_better=False)

    selected.fit(x,y)
    y_pred = selected.predict(x)
    accuracy = metric(y, y_pred)

    print(selected)
    print('Accuracy in the trainning set: {}'.format(accuracy))