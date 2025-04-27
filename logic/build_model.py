import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor

def build_model():
    cb_model = CatBoostRegressor(
        iterations=500,
        border_count=32,
        depth=8,
        l2_leaf_reg=1,
        learning_rate=0.01,
        verbose=False
    )
    print("model builded successfully!")
    return cb_model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)

def predict(model, X_test):
    return model.predict(X_test)

def calculate_rmse(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, np.round(y_pred)))

def map_anxiety_level(score):
    if 1 <= score <= 3:
        return 'Low Anxiety'
    elif 4 <= score <= 6:
        return 'Moderate Anxiety'
    elif 7 <= score <= 8:
        return 'High Anxiety'
    elif 9 <= score <= 10:
        return 'Very High Anxiety'
    else:
        return 'Unknown'

def predict_and_map_anxiety(model, X_test):

    predicted_scores = predict(model, X_test)

    rounded_scores = np.round(predicted_scores)

    predicted_labels = [map_anxiety_level(score) for score in rounded_scores]

    return predicted_scores, rounded_scores, predicted_labels
