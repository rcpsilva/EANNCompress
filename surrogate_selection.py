import random
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

def rand(surrogate_ensemble,x,y):
    return random.choice(surrogate_ensemble)

def by_metric(surrogate_ensemble,metric,x,y):
    """ Select the best surrogate using the input metric
    """
    pass

#### Example 
if __name__ == "__main__":

    surrogate_ensemble = [DecisionTreeRegressor(),
        LinearRegression(),
        KNeighborsRegressor()]

    x = np.random.rand(100,3) * 10
    y = x[:,0]**2 + x[:,1]**2 + x[:,2]**2

    metric = mean_squared_error

    selected = by_metric(surrogate_ensemble,metric,x,y)

    selected.fit(x,y)
    y_pred = selected.predict(x)
    accuracy = metric(y, y_pred)

print(selected)
print('Accuracy in the trainning set: {}'.format(accuracy))