from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

dataset = load_boston()
data = pd.DataFrame(data= dataset.data, columns=dataset.feature_names)
features = data.drop(['INDUS', 'AGE'], axis=1)

log_prices = np.log(dataset.target)
target = pd.DataFrame(log_prices, columns=['PRICE'])

property_stats = np.ndarray(shape=(1,11))
property_stats = features.mean().values.reshape(1, 11)

regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)

RM_IDX = 4
PTRATIO_IDX = 8
CHAS_IDX = 2

MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)

def get_log_estimate(nr_rooms, students_per_classroom, next_to_river=False, high_confidence=True):
   
    # configure property
    property_stats[0][RM_IDX] = nr_rooms
    property_stats[0][PTRATIO_IDX] = students_per_classroom
    
    if next_to_river:
        property_stats[0][CHAS_IDX] = 1
    else:
        property_stats[0][CHAS_IDX] = 0
    
    # Make prediction
    log_estimate = regr.predict(property_stats)[0][0]
    
    # calc range
    if high_confidence:
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval = 95
    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68
        
    return log_estimate, upper_bound, lower_bound, interval

ZILLOW_MEDIAN_PRICE = 583.3
SCALE_FACTOR = ZILLOW_MEDIAN_PRICE / np.median(dataset.target)

log_est, upper, lower, conf = get_log_estimate(9, students_per_classroom=15, next_to_river=False, high_confidence=False)

dollar_est = np.e ** log_est * 1000 * SCALE_FACTOR
dollar_high = np.e ** upper * 1000 * SCALE_FACTOR
dollar_low = np.e ** lower * 1000 * SCALE_FACTOR

rounded_est = np.around(dollar_est, -3)
rounded_high = np.around(dollar_high, -3)
rounded_low = np.around(dollar_low, -3)

print(f'Price in Dollars: {rounded_est}')
print(f'Upper: {rounded_high}')
print(f'Lower: {rounded_low}')
print(f'Confidence: {conf}')

def get_dollar_estimate(rm, ptratio, chas=False, large_range=True):
    
    """Estimate the price of a property in Boston.
    
    Keyword arguments:
    rm -- number of rooms in the property.
    ptratio -- number of students per teacher in the classroom for the school in the area.
    chas -- True if the property is next to the river, False otherwise.
    large_range -- True for a 95% prediction interval, False for a 68% interval.
    
    """
    
    
    if rm < 1 or ptratio < 1:
        print('That is unrealistic. Try again.')
        return
    
    log_est, upper, lower, conf = get_log_estimate(rm, ptratio, chas, large_range)
    
    dollar_est = np.e ** log_est * 1000 * SCALE_FACTOR
    dollar_high = np.e ** upper * 1000 * SCALE_FACTOR
    dollar_low = np.e ** lower * 1000 * SCALE_FACTOR

    rounded_est = np.around(dollar_est, -3)
    rounded_high = np.around(dollar_high, -3)
    rounded_low = np.around(dollar_low, -3)

    print(f'Price in Dollars: {rounded_est}')
    print(f'Upper: {rounded_high}')
    print(f'Lower: {rounded_low}')
    print(f'Confidence: {conf}%')