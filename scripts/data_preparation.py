import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer


class Data_Preparation:
    # Load dataset
    df = pd.read_csv("store_sales.csv")

    # ----- Handling Non-Numeric Columns -----
    # Example: Encoding StoreType, Assortment, and StateHoliday using one-hot encoding
    df = pd.get_dummies(df, columns=['StoreType', 'Assortment', 'StateHoliday'], drop_first=True)

    # ----- Handling Missing Values -----
    # Impute missing values (mean for numeric, most frequent for categorical)
    imputer_numeric = SimpleImputer(strategy='mean')
    imputer_categorical = SimpleImputer(strategy='most_frequent')

    # Apply imputer for numeric and categorical columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    df[numeric_cols] = imputer_numeric.fit_transform(df[numeric_cols])
    df[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])

    # ----- Generating New Features -----
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract date-based features
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['weekofyear'] = df['Date'].dt.isocalendar().week

    # Create a feature for beginning, mid, or end of the month
    df['month_period'] = pd.cut(df['Date'].dt.day, bins=[0, 10, 20, 31], labels=['begin', 'mid', 'end'])

    # Scaling features using StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[numeric_cols])

    # Updating the DataFrame with scaled values
    df[numeric_cols] = scaled_features
