import pandas as pd
import numpy as np
import math
import lightgbm as lgb
from sklearn.preprocessing import QuantileTransformer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import streamlit as st
import folium
from folium import Marker
from streamlit_folium import folium_static

# Data processing
data = pd.read_csv("new.csv")

# Model function
def Dragon(data, lat, long):
    X = data.drop(columns=['latitude', 'longitude'])
    scaler = QuantileTransformer(output_distribution='uniform')
    X = scaler.fit_transform(X)
    y1 = data.latitude
    y2 = data.longitude
    y = np.column_stack((y1, y2))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.1
    }

    lgb_regressor = lgb.LGBMRegressor(**params)
    multi_output_regressor = MultiOutputRegressor(lgb_regressor)
    multi_output_regressor.fit(X_train, y_train)

    multi_output_regressor.estimators_[0].booster_.save_model('model_latitude.txt')
    multi_output_regressor.estimators_[1].booster_.save_model('model_longitude.txt')

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.01,
        'min_gain_to_split': 0,
        'min_data_in_leaf': 1,
        'num_leaves': 100,
        'max_depth': -1
    }
    y1 = np.array(y1)
    y2 = np.array(y2)
    X = np.array(X)
    y_new = []
    X_new = []
    for i in range(len(y1)):
        k = math.sqrt(math.pow(y1[i] - long, 2) + math.pow(y2[i] - lat, 2))
        if 2 < k <= 5:
            y_new.append(y[i])
            X_new.append(X[i])
    y_new = np.array(y_new)
    X_new = np.array(X_new)
    Xtestnew = []
    for i in range(len(y1)):
        k = math.sqrt(math.pow(y1[i] - long, 2) + math.pow(y2[i] - lat, 2))
        if k < 2:
            Xtestnew.append(X[i])
    Xtestnew = np.array(Xtestnew)
    lgb_regressor_latitude = lgb.Booster(model_file='model_latitude.txt')
    lgb_regressor_longitude = lgb.Booster(model_file='model_longitude.txt')

    lgb_regressor_latitude = lgb.train(params, lgb.Dataset(X_new, label=y_new[:, 0]),
                                       num_boost_round=50, init_model=lgb_regressor_latitude)
    lgb_regressor_longitude = lgb.train(params, lgb.Dataset(X_new, label=y_new[:, 1]),
                                        num_boost_round=50, init_model=lgb_regressor_longitude)

    class CustomMultiOutputRegressor:
        def __init__(self, regressor_latitude, regressor_longitude):
            self.regressor_latitude = regressor_latitude
            self.regressor_longitude = regressor_longitude

        def predict(self, X):
            pred_latitude = self.regressor_latitude.predict(X, num_iteration=self.regressor_latitude.best_iteration)
            pred_longitude = self.regressor_longitude.predict(X, num_iteration=self.regressor_longitude.best_iteration)
            return np.column_stack((pred_longitude, pred_latitude))

    custom_multi_output_regressor = CustomMultiOutputRegressor(lgb_regressor_latitude, lgb_regressor_longitude)
    y_pred = custom_multi_output_regressor.predict(Xtestnew)

    ans = y_pred
    ans1 = ans[:, 0]
    ans2 = ans[:, 1]
    y1_nw = []
    for i in range(len(ans1)):
        k = math.sqrt(math.pow(ans2[i] - long, 2) + math.pow(ans1[i] - lat, 2))
        if k <= 1.5:
            y1_nw.append(ans[i])
    y1_nw = np.array(y1_nw)
    return y1_nw

lat = -101
long = 38
model_output = Dragon(data, lat, long)

# Function to generate the map with markers
def create_map(coordinates):
    if len(coordinates) > 0:
        initial_location = coordinates[0]
    else:
        initial_location = [0, 0]

    m = folium.Map(location=initial_location, zoom_start=5)
    for coord in coordinates:
        Marker(location=coord).add_to(m)
    return m

# Streamlit app
st.title("Location Visualization")

st.markdown("""
    This app displays the coordinates output by the model on an OpenStreetMap.
""")
coordinates = [tuple(coord) for coord in model_output]

map_object = create_map(coordinates)
folium_static(map_object)
