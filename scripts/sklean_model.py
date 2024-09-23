import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Model:
    def __init__(self) -> None:
        logger.info("Model class initialized.")

    # ----- Build Model Pipeline -----
    def model(self, data: pd.DataFrame, preprocessor) -> Pipeline:
        logger.info("Preparing the data and creating the model pipeline.")
        
        # Splitting Data into Features and Target
        X = data.drop(columns=['Sales', 'Date'])  # Features (exclude 'Sales' and 'Date')
        y = data['Sales']  # Target variable (Sales)

        # Split the data into training and testing sets (80/20 split)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the Pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),  # Apply preprocessing (scaling and encoding)
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))  # Random Forest Model
        ])

        logger.info("Pipeline successfully created.")
        return pipeline, X_train, X_test, y_train, y_test

    # ----- Train the Model -----
    def model_train(self, X_train: pd.DataFrame, y_train: pd.Series, pipeline: Pipeline) -> Pipeline:
        logger.info("Training the model.")
        pipeline.fit(X_train, y_train)
        logger.info("Model training complete.")
        return pipeline

    # ----- Cross-Validation -----
    def cross_val(self, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        logger.info("Performing cross-validation.")
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)  # Convert negative MSE to RMSE
        mean_cv_rmse = cv_rmse.mean()
        logger.info(f"Cross-Validation RMSE: {mean_cv_rmse:.4f}")
        return mean_cv_rmse

    # ----- Make Predictions -----
    def prediction(self, X_test: pd.DataFrame, pipeline: Pipeline) -> np.ndarray:
        logger.info("Making predictions on the test data.")
        y_pred = pipeline.predict(X_test)
        logger.info("Predictions completed.")
        return y_pred

    # ----- Evaluate Model -----
    def evaluate_model(self, y_test: pd.Series, y_pred: np.ndarray) -> dict:
        logger.info("Evaluating the model.")
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info(f"Test RMSE: {test_rmse:.4f}")
        logger.info(f"Test MAE: {mae:.4f}")
        logger.info(f"R^2 Score: {r2:.4f}")

        return {'rmse': test_rmse, 'mae': mae, 'r2': r2}

    # ----- Huber Loss -----
    def huber_loss(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        logger.info("Calculating Huber Loss.")
        huber = HuberRegressor()
        huber.fit(X_train, y_train)
        huber_pred = huber.predict(X_test)

        huber_loss_value = mean_squared_error(y_test, huber_pred)
        logger.info(f"Huber Loss (approximated by MSE): {huber_loss_value:.4f}")
        
        return huber_loss_value
    
    # Post-prediction analysis
    def post_prediction_analysis(self, y_test: pd.Series, y_pred: np.ndarray) -> None:
        logger.info("Performing post-prediction analysis.")

        residuals = y_test - y_pred

        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=y_pred, y=residuals)
        plt.axhline(0, color='red', linestyle='--', lw=2)
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Predicted Values")
        plt.show()

        plt.figure(figsize=(8, 5))
        sns.histplot(residuals, kde=True)
        plt.title("Distribution of Residuals")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.show()

    # Feature Importance (for RandomForestRegressor)
    def feature_importance(self, pipeline: Pipeline, feature_names: list) -> None:
        logger.info("Analyzing feature importance for RandomForestRegressor.")
        model = pipeline.named_steps['model']

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)

            plt.figure(figsize=(10, 5))
            plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx], color='teal')
            plt.title("Feature Importance")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.show()

        else:
            logger.warning("The model does not support feature importance evaluation.")

    # Save Model
    def save_model(self, pipeline: Pipeline, filepath: str) -> None:
        logger.info(f"Saving the model to {filepath}")
        joblib.dump(pipeline, filepath)
        logger.info(f"Model saved successfully to {filepath}")

    # Load Model
    def load_model(self, filepath: str) -> Pipeline:
        logger.info(f"Loading model from {filepath}")
        pipeline = joblib.load(filepath)
        logger.info(f"Model loaded successfully from {filepath}")
        return pipeline