import joblib
import pandas as pd
from predicting_social_anxiety.logic.preprocessor import build_pipeline
from predicting_social_anxiety.logic.data import load_data, cat_num, split
from predicting_social_anxiety.logic.build_model import build_model, train_model, predict, calculate_rmse, map_anxiety_level, predict_and_map_anxiety

def main():
    csv_path = '/home/asal/code/AsalSaud/predicting_social_anxiety/raw_data/enhanced_anxiety_dataset.csv'
    X, y, df= load_data(csv_path)

    # preprocess data
    num, cat = cat_num(X)
    preprocessor = build_pipeline(cat)

    X_ohe = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    X_ohe = pd.DataFrame(X_ohe, columns=feature_names)
    X_merged= pd.concat([X_ohe, X[num]], axis=1)

    # Split data in train and test set
    X_train, X_test, y_train, y_test = split(X_merged, y)

    # Build and train model
    cb_model = build_model()
    train_model(cb_model, X_train, y_train)

    # Make predictions
    y_pred = predict(cb_model, X_test)

    # Evaluate model
    rmse_ = calculate_rmse(y_test, y_pred)
    print(f"Root Mean Squared Error (RMSE): {rmse_}")

    predicted_scores, rounded_scores, predicted_labels = predict_and_map_anxiety(cb_model, X_test)

    print(f"You got {predicted_scores[0]} which indicates a {map_anxiety_level(predicted_scores[0])}")

    # # Save the model
    joblib.dump(cb_model, 'cb2_model.joblib')
    joblib.dump(preprocessor, 'cb2_preprocessor.joblib')

    print("everything is good :)")


if __name__ == "__main__":
    main()
