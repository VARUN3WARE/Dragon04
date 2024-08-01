import pandas as pd
import numpy as np
import math
import lightgbm as lgb
from sklearn.preprocessing import QuantileTransformer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import streamlit as st
import folium
from streamlit_folium import st_folium

# Data processing
data = pd.read_csv("new.csv")

# Model function
def Dragon(data, lat, long):
    # Swap latitude and longitude
    lat, long = long, lat

    # Feature processing
    X = data.drop(columns=['latitude', 'longitude'])
    scaler = QuantileTransformer(output_distribution='uniform')
    X = scaler.fit_transform(X)
    y1 = data.latitude
    y2 = data.longitude
    y = np.column_stack((y1, y2))

    # Initial model training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    params = {'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.1}
    lgb_regressor = lgb.LGBMRegressor(**params)
    multi_output_regressor = MultiOutputRegressor(lgb_regressor)
    multi_output_regressor.fit(X_train, y_train)
    
    # Save the initial model
    multi_output_regressor.estimators_[0].booster_.save_model('model_latitude.txt')
    multi_output_regressor.estimators_[1].booster_.save_model('model_longitude.txt')

    # Filter data based on distance
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
        if k < 3:
            Xtestnew.append(X[i])
    Xtestnew = np.array(Xtestnew)

    # Retrain the model with filtered data
    lgb_regressor_latitude = lgb.Booster(model_file='model_latitude.txt')
    lgb_regressor_longitude = lgb.Booster(model_file='model_longitude.txt')
    params.update({'learning_rate': 0.01, 'min_gain_to_split': 0, 'min_data_in_leaf': 1, 'num_leaves': 100, 'max_depth': -1})
    try:
        lgb_regressor_latitude = lgb.train(params, lgb.Dataset(X_new, label=y_new[:, 0]), num_boost_round=50, init_model=lgb_regressor_latitude)
        lgb_regressor_longitude = lgb.train(params, lgb.Dataset(X_new, label=y_new[:, 1]), num_boost_round=50, init_model=lgb_regressor_longitude)

        # Custom wrapper for predictions
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

        # Filter predictions based on distance
        ans1 = y_pred[:, 0]
        ans2 = y_pred[:, 1]
        y1_nw = []
        for i in range(len(ans1)):
            k = math.sqrt(math.pow(ans2[i] - long, 2) + math.pow(ans1[i] - lat, 2))
            if k <= 2:
                y1_nw.append(y_pred[i])
        y1_nw = np.array(y1_nw)
        return y1_nw
    except Exception as e:
        print(e)
        return np.array([])

# Streamlit app setup
st.title("Interactive World Map with Clickable Points")

# Initialize Folium map
map_center = [37.0, -101.0]
m = folium.Map(location=map_center, zoom_start=6)

# Add initial marker
click_marker = folium.Marker(location=map_center, draggable=False, icon=folium.Icon(color='red'))
click_marker.add_to(m)

# Handle map clicks
def map_click(lat, lon):
    try:
        model_output = Dragon(data, lat, lon)
        model_output = np.array(model_output)
        st.write(f"Number of locations within the range: {len(model_output)}")
        for loc in model_output:
            loc = loc.tolist()
            loc[0], loc[1] = loc[1], loc[0]
            st.markdown(f"{loc}")
            if isinstance(loc, (list, np.ndarray)) and len(loc) == 2:
                loc_tuple = tuple(loc)  # Convert to tuple
                st.write(f"Adding marker at: {loc_tuple}")
                folium.Marker(loc_tuple, popup=f"Coordinates: {lat}, {lon}").add_to(m)
            else:
                st.write(f"Invalid location format: {loc}")

    except Exception as e:
        st.write(f"Error: {e}")

# Display the map and handle click events
output = st_folium(m, width=700, height=500)
if output['last_clicked'] is not None:
    clicked_lat = output['last_clicked']['lat']
    clicked_lon = output['last_clicked']['lng']
    map_click(clicked_lat, clicked_lon)
    st.write(f"Clicked coordinates: Latitude = {clicked_lat}, Longitude = {clicked_lon}")

# Re-render the map with the updated markers
output = st_folium(m, width=700, height=500)
