import random
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

def select_random(surrogate_ensemble,x,y):
    return random.choice(surrogate_ensemble)

def select_by_MSE(surrogate_ensemble,x,y):
    """ Select the best surrogate using the Mean Absolute Error
    """
    pass

def select_by_MAE(surrogate_ensemble,x,y):
    """ Select the best surrogate using the Mean Absolute Error
    """
    pass

def select_by_MAPE(surrogate_ensemble,x,y):
    """ Select the best surrogate using the Mean Absolute Percentual Error
    """
    pass

def select_by_Spearman(surrogate_ensemble,x,y):
    """ Select the best surrogate using Sepearman's rank correlation
    """
    pass

#### Example 
surrogate_ensemble = [DecisionTreeRegressor(),
    LinearRegression(),
    KNeighborsRegressor()]

x = np.random.rand(100,3) * 10
y = x[:,0]**2 + x[:,1]**2 + x[:,2]**2

selected = select_by_MSE(surrogate_ensemble,x,y)

selected.fit(x,y)
y_pred = selected.predict(x)
accuracy = mean_squared_error(y, y_pred)

print(selected)
print('Accuracy in the trainning set: {}'.format(accuracy))