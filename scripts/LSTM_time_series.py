import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam


# ----- Step 1: Data Preprocessing -----

class LSTMModel:
    
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
    
    def date_conversion(self,df):
        # Convert 'Date' column to datetime format
        df['Date'] = pd.to_datetime(df['Date'])

        # Sort the data by the 'Date' column in ascending order
        df = df.sort_values(by='Date', ascending=True)

        # Reset index after sorting
        df = df.reset_index(drop=True)
        return df
    
    # Function to check stationarity using the Augmented Dickey-Fuller test
    def check_stationarity(self, data):
        result = adfuller(data['Sales'])
        print('ADF Statistic:', result[0])
        print('p-value:', result[1])
        if result[1] <= 0.05:
            print("Data is stationary")
        else:
            print("Data is non-stationary. Differencing needed.")
    
    # Create lagged features for supervised learning
    def create_lagged_features(self, data, n_lags=7):
        df = data.copy()
        for i in range(1, n_lags + 1):
            df[f"lag_{i}"] = df['Sales'].shift(i)
        df.dropna(inplace=True)
        return df
    
    # Scaling data using MinMaxScaler
    def scale_data(self, data):
        # Columns to scale
        columns_to_scale = ['Customers', 'CompetitionDistance', 'CompetitionOpenTime', 'Promo2OpenTime']
        target_column = ['Sales']
        # Initialize the scaler
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Fit and transform the features
        data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

        # Fit and transform the target
        unscaled_target=data[target_column]
        scaled_target = scaler.fit_transform(data[target_column])
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
        return scaler,scaled_target,unscaled_target,data
    
    def managinig_data_types(self,df):
        # One-hot encoding categorical features
        df = pd.get_dummies(df, columns=['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval'])
        boll_df=df.select_dtypes("bool")
        bool_col=boll_df.columns
        for col in bool_col:
            df[col] = df[col].astype(int)
        return df
    
    # Function to split the data into sequences (for LSTM input)
    def create_sequences(self,data, n_lag):
        X, y = [], []
        for i in range(len(data) - n_lag):
            X.append(data[i:i + n_lag])
            y.append(data[i + n_lag, 0])  # Assuming target is 'Sales'
        return np.array(X), np.array(y)


    # ---------- Split the data into training and test sets-----------
    def splite_data(self, X,y):
        # Step 1: Define the split ratio
        train_size = int(len(X) * 0.8)  # 80% for training

        
        X_train, X_test = X[:train_size], X[train_size:]  # First 80% for training, last 20% for testing
        y_train, y_test = y[:train_size], y[train_size:]

        # Check the shapes
        print("Training data shape:", X_train.shape, y_train.shape)
        print("Testing data shape:", X_test.shape, y_test.shape)

        return X_train,y_train, X_test,y_test

    #--------- Building LSTM Model -----
    def build_lstm_model(self, n_timesteps, n_features):
        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(n_timesteps, n_features)))
        model.add(LSTM(units=50))
        model.add(Dense(1))  # Output layer for regression

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    

