import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import logging
import datetime
from dateutil.easter import easter

import seaborn as sns
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO)
class DataProcess:
    def __init__(self):

        pass

    def merging_store(selff,store,train,test):
        # Converting 'Date' to datetime in both train and test sets
        train['Date'] = pd.to_datetime(train['Date'])
        test['Date'] = pd.to_datetime(test['Date'])

        # Handling missing values
        store['CompetitionDistance'] = store['CompetitionDistance'].fillna(store['CompetitionDistance'].median())
        store['CompetitionOpenSinceMonth'] = store['CompetitionOpenSinceMonth'].fillna(0)
        store['CompetitionOpenSinceYear'] = store['CompetitionOpenSinceYear'].fillna(0)
        store['Promo2SinceWeek'] = store['Promo2SinceWeek'].fillna(0)
        store['Promo2SinceYear'] = store['Promo2SinceYear'].fillna(0)
        store['PromoInterval'] = store['PromoInterval'].fillna('None')

        # Convert numerical values to strings for the entire 'StateHoliday' column
        train["StateHoliday"] = train["StateHoliday"].astype(str)
        
        # Merging Store info with Train and Test
        test["Open"] = test['Open'].ffill()
        train = train.merge(store, on='Store', how='left')
        test = test.merge(store, on='Store', how='left')
        return store,train, test

    def feature_engineering(self, train, test):
        # Featuring Engineering - Date Features
        train['Month'] = train['Date'].dt.month
        train['Year'] = train['Date'].dt.year
        train['Day'] = train['Date'].dt.day
        train['WeekOfYear'] = train['Date'].dt.isocalendar().week
        train['is_weekend'] = train['DayOfWeek'].apply(lambda x: 1 if x in [6,7] else 0)
        test['Month'] = test['Date'].dt.month
        test['Year'] = test['Date'].dt.year
        test['Day'] = test['Date'].dt.day
        test['WeekOfYear'] = test['Date'].dt.isocalendar().week
        test['is_weekend'] = test['DayOfWeek'].apply(lambda x: 1 if x in [6,7] else 0)
        # Impute missing values (mean for numeric, most frequent for categorical)
        imputer_numeric = SimpleImputer(strategy='mean')
        imputer_categorical = SimpleImputer(strategy='most_frequent')
        
        # Convert numerical values to strings for the entire 'StateHoliday' column
        train["StateHoliday"] = train["StateHoliday"].astype(str)
        # Apply imputer for numeric and categorical columns
        numeric_cols = train.select_dtypes(include=np.number).columns
        categorical_cols = train.select_dtypes(include=['object', 'category']).columns
       
        train[numeric_cols] = imputer_numeric.fit_transform(train[numeric_cols])
        train[categorical_cols] = imputer_categorical.fit_transform(train[categorical_cols])
        # Creating promo and competition features
        # Competition Open time
        train['CompetitionOpenTime'] = 12 * (train['Year'] - train['CompetitionOpenSinceYear']) + \
                                        (train['Month'] - train['CompetitionOpenSinceMonth'])
        test['CompetitionOpenTime'] = 12 * (test['Year'] - test['CompetitionOpenSinceYear']) + \
                                        (test['Month'] - test['CompetitionOpenSinceMonth'])

        # Promo2 Open time
        train['Promo2OpenTime'] = 12 * (train['Year'] - train['Promo2SinceYear']) + \
                                (train['WeekOfYear'] - train['Promo2SinceWeek']) / 4.0
        test['Promo2OpenTime'] = 12 * (test['Year'] - test['Promo2SinceYear']) + \
                                (test['WeekOfYear'] - test['Promo2SinceWeek']) / 4.0

        train['Promo2OpenTime'] = train['Promo2OpenTime'].apply(lambda x: 0 if x < 0 else x)
        test['Promo2OpenTime'] = test['Promo2OpenTime'].apply(lambda x: 0 if x < 0 else x)

        # Encoding Categorical Columns
        categorical_cols = ['StoreType', 'Assortment', 'StateHoliday', 'PromoInterval']

        # Create a preprocessor pipeline for categorical and numerical features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['CompetitionDistance', 'CompetitionOpenTime', 'Promo2OpenTime', 'DayOfWeek']),
                ('cat', OneHotEncoder(), categorical_cols)
            ])
        
        return train, test, preprocessor


    
    def prom_distribution(self, train, test):
        # Add a column to distinguish between train and test datasets
        train['Dataset'] = 'Train'
        test['Dataset'] = 'Test'
        
        # Concatenate both datasets
        combined_data = pd.concat([train[['Promo', 'Dataset']], test[['Promo', 'Dataset']]])

        # Calculate the count of promos for each dataset
        promo_count = combined_data.groupby(['Dataset', 'Promo']).size().reset_index(name='Count')

        # Calculate the total count for each dataset
        total_count = promo_count.groupby('Dataset')['Count'].sum().reset_index(name='Total')

        # Merge back to get proportions
        promo_proportion = pd.merge(promo_count, total_count, on='Dataset')
        promo_proportion['Proportion'] = 100 * promo_proportion['Count'] / promo_proportion['Total']

        # Plot the proportions
        plt.figure(figsize=(10,6))
        sns.barplot(x='Promo', y='Proportion', hue='Dataset', data=promo_proportion)
        plt.title('Proportion of Promotions in Train and Test Sets')
        plt.xlabel('Promo')
        plt.ylabel('Proportion (%)')
        plt.show()




    def sales_behaviorOnholiday(self, train):
        # Convert 'Date' to datetime
        train['Date'] = pd.to_datetime(train['Date'])

        # Filter for holidays
        holiday_sales = train[train['StateHoliday'] != '0'].copy()  # Use .copy() to avoid SettingWithCopyWarning

        # Map 'StateHoliday' values to meaningful names
        holiday_mapping = {
            'a': 'Public Holiday',
            'b': 'Easter Holiday',
            'c': 'Christmas Holiday'
        }

        holiday_sales.loc[:, 'StateHoliday'] = holiday_sales['StateHoliday'].map(holiday_mapping)

        # Filter out any rows where 'StateHoliday' is still null after mapping
        holiday_sales = holiday_sales[holiday_sales['StateHoliday'].notnull()]

        # Sales behavior before, during, and after holidays
        plt.figure(figsize=(12,6))
        sns.lineplot(x='Date', y='Sales', data=holiday_sales, hue='StateHoliday')
        plt.title('Sales Before, During, and After Holidays')
        plt.show()



    def Seasonal_Purchase_Behaviors(self, train):
        # Convert 'Date' to datetime
        train['Date'] = pd.to_datetime(train['Date'])

        # Map 'StateHoliday' values to meaningful names
        holiday_mapping = {
            'a': 'Public Holiday',
            'b': 'Easter Holiday',
            'c': 'Christmas Holiday'
        }

        train['StateHoliday'] = train['StateHoliday'].map(holiday_mapping)

        # Filter for holidays (now using the descriptive names)
        holiday_sales = train[train['StateHoliday'].notnull()]  # Excludes rows where it's not a holiday

        # Barplot showing the average sales for each holiday type
        plt.figure(figsize=(12,6))
        sns.barplot(x='StateHoliday', y='Sales', hue='StateHoliday', data=holiday_sales,
                palette=['#87cefa', '#ffa07a', '#20b2aa'], legend=False, errorbar=None)
        plt.title('Average Sales for Different Holidays')
        plt.xlabel('Holiday Type')
        plt.ylabel('Average Sales')

        plt.show()


    def sales_customer_correlation(self, train):
        # Correlation between sales and customers
        corr = train['Sales'].corr(train['Customers'])
        print(f"Correlation between Sales and Customers: {corr}")

        # Scatter plot
        plt.figure(figsize=(12,6))
        sns.scatterplot(x='Customers', y='Sales', data=train)
        plt.title('Sales vs. Number of Customers')
        plt.show()

    
    def Promotion_effect_on_sales_monthly(self, train):
        # Convert 'Date' column to datetime if not already
        train['Date'] = pd.to_datetime(train['Date'])
        
        # Add a new column 'Month' to represent the year and month
        train['Month'] = train['Date'].dt.to_period('M')

        # Group data by 'Month' and 'Promo' and calculate the mean sales for each month
        monthly_sales = train.groupby(['Month', 'Promo'])['Sales'].mean().unstack()

        # Plot sales over time (monthly)
        plt.figure(figsize=(14,8))
        
        # Plot sales with promo
        plt.plot(monthly_sales.index.to_timestamp(), monthly_sales[1], 
                label='Sales with Promo', color='green', marker='o')
        
        # Plot sales without promo
        plt.plot(monthly_sales.index.to_timestamp(), monthly_sales[0], 
                label='Sales without Promo', color='blue', marker='o')
        
        # Add labels and title
        plt.title('Monthly Average Sales with and without Promotion (2013-2015)', fontsize=16)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Average Sales', fontsize=12)
        
        # Show legend
        plt.legend()
        
        # Display the plot
        plt.show()


    def promos_deployement(self, train):
        # Ensure 'StoreType' and 'Promo' are treated as categorical variables
        train['StoreType'] = train['StoreType'].astype('category')
        train['Promo'] = train['Promo'].astype('category')

        # Promo effectiveness by store type
        plt.figure(figsize=(12,6))
        sns.boxplot(x='StoreType', y='Sales', hue='Promo', data=train)
        plt.title('Promo Effectiveness by Store Type')
        plt.xlabel('Store Type')
        plt.ylabel('Sales')
        plt.show()


    def store_closing_times(self, train):
        # Map Open column to 'Closed' and 'Open' strings
        train['Open'] = train['Open'].map({0: 'Closed', 1: 'Open'})

        # Convert 'Open' to categorical type to avoid the warning
        train['Open'] = train['Open'].astype('category')

        # Customer behavior during store open and close
        plt.figure(figsize=(12,6))
        sns.boxplot(x='Open', y='Customers', data=train)
        plt.title('Customer Behavior During Store Open and Close')
        plt.xlabel('Store Status')
        plt.ylabel('Number of Customers')
        plt.show()


    def store_on_holiday(self, train):
        # Group by stores and weekday sales
        weekday_sales = train.groupby(['Store', 'DayOfWeek'])['Sales'].mean().reset_index()

        # Line plot
        plt.figure(figsize=(12,6))
        sns.lineplot(x='DayOfWeek', y='Sales', hue='Store', data=weekday_sales)
        plt.title('Sales Trends Over Weekdays and Weekends')
        plt.show()

    def competitor_affect_on_sales(self,train):
        # Scatter plot of competition distance vs sales
        plt.figure(figsize=(12,6))
        sns.scatterplot(x='CompetitionDistance', y='Sales', data=train)
        plt.title('Competition Distance vs Sales')
        plt.show()
    

    def competitor_affect_on_sales_line(self, train):
        # Create bins for 'CompetitionDistance'
        bins = pd.cut(train['CompetitionDistance'], bins=10)  # Adjust the number of bins if necessary

        # Group by the bins and calculate the mean sales for each bin
        binned_sales = train.groupby(bins)['Sales'].mean()

        # Plot sales vs competition distance bins
        plt.figure(figsize=(12,6))
        
        # Plot the average sales for each competition distance bin
        plt.plot(binned_sales.index.astype(str), binned_sales, marker='o', color='orange')
        
        # Add labels and title
        plt.title('Average Sales Across Competition Distance Bins', fontsize=16)
        plt.xlabel('Competition Distance Bins', fontsize=12)
        plt.ylabel('Average Sales', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Display the plot
        plt.show()


    def Reopening_New_Competitors_affects(self, store,train):
        # Fill missing values in competition distance
        store['CompetitionDistance'] = store['CompetitionDistance'].fillna(0)

        # Analyze stores with NA to filled values
        affected_stores = store[store['CompetitionDistance'] > 0]

        # Plot sales before and after competition
        plt.figure(figsize=(12,6))
        sns.lineplot(x='Date', y='Sales', data=train[train['Store'].isin(affected_stores['Store'])])
        plt.title('Sales Before and After Competitor Opening')
        plt.show()

    def plot_sales_vs_customers(self,train):
        plt.figure(figsize=(10,6))
        sns.scatterplot(x='Customers', y='Sales', data=train)
        plt.title('Sales vs Customers')
        plt.xlabel('Customers')
        plt.ylabel('Sales')
        plt.show()

    def stores_weekday_sales (self, train):
        # Group by stores and weekday sales
        weekday_sales = train.groupby(['Store', 'DayOfWeek'])['Sales'].mean().reset_index()

        # Line plot
        plt.figure(figsize=(12,6))
        sns.lineplot(x='DayOfWeek', y='Sales', hue='Store', data=weekday_sales)
        plt.title('Sales Trends Over Weekdays and Weekends')
        plt.show()
    
    def self_assortment_type(self, train):

        # Sales by assortment type

        assortment_mapping={
            'a' : "basic", 
            'b': "extra", 
            'c' :"extended"
        }
        train.loc[:, 'Assortment'] = train['Assortment'].map(assortment_mapping)
        plt.figure(figsize=(12,6))
        sns.boxplot(x='Assortment', y='Sales', data=train)
        plt.title('Sales by Assortment Type')
        plt.show()



    def plot_sales_during_holidays(self,train):
        plt.figure(figsize=(10,6))
        sns.boxplot(x='StateHoliday', y='Sales', data=train)
        plt.title('Sales During State Holidays')
        plt.xlabel('State Holiday')
        plt.ylabel('Sales')
        plt.show()

        plt.figure(figsize=(10,6))
        sns.boxplot(x='SchoolHoliday', y='Sales', data=train)
        plt.title('Sales During School Holidays')
        plt.xlabel('School Holiday')
        plt.ylabel('Sales')
        plt.show()

    def plot_sales_with_promo(self,train):
        plt.figure(figsize=(10,6))
        sns.boxplot(x='Promo', y='Sales', data=train)
        plt.title('Sales with and without Promo')
        plt.xlabel('Promo')
        plt.ylabel('Sales')
        plt.show()
    

    def plot_sales_seasonality(self, train):
        # Ensure 'Date' column is in datetime format
        train['Date'] = pd.to_datetime(train['Date'])
        
        # Extract month from the 'Date' column
        train['Month'] = train['Date'].dt.month

        # Group by 'Month' and calculate the average sales for each month
        monthly_sales = train.groupby('Month')['Sales'].mean()

        # Plot the monthly sales trends
        plt.figure(figsize=(14,8))
        monthly_sales.plot(kind='line', marker='o', color='green')
        plt.title('Average Sales by Month (Seasonality Trends)')
        plt.xlabel('Month')
        plt.ylabel('Average Sales')
        plt.xticks(range(1, 13))  # Set x-ticks to represent months (1-12)
        plt.grid(True)
        plt.show()


    def save_processed_data_to_csv(self,processed_data, filename):
        # Save the DataFrame to a CSV file
        processed_data.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        






