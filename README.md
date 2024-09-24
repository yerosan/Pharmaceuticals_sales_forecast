## Pharmaceuticals Sales Forecasting
This repository contains the code and notebooks for forecasting pharmaceutical sales using machine learning models like LSTM and Scikit-learn. 

The project structure is organized to streamline data processing, model building, and performance evaluation.

### Project Structure

```bash
.
├── Data                      # Contains the dataset for analysis and modeling
├── notebooks
│   ├── exploratory_data_analysis.ipynb   # Initial EDA for insights
│   ├── lstm_model.ipynb                   # LSTM model for time-series forecasting
│   └── model.ipynb                        # Additional model exploration (Sklearn, etc.)
├── result                    # Placeholder for model results and outputs
├── scripts
│   ├── data_accessing.py                 # Script to access the dataset
│   ├── data_preparation.py               # Handles data cleaning and feature engineering
│   ├── data_preprocessing.py             # Preprocessing for time-series data
│   ├── LSTM_time_series.py               # LSTM model training script
│   └── sklearn_model.py                  # Sklearn-based model script
├── src
│   └── app.py                            #  FasAPI app for running the forecasting service
├── tests                    # Unit tests for the codebase
└── README.md                # Project documentation
```
### Model Findings
**LSTM Model:** Achieved promising results in capturing sales trends, especially long-term patterns.

**Scikit-learn Models:** Provided baseline predictions, with room for improvement using advanced features.

**Evaluation Metrics:** The models are evaluated using RMSE, MSE, and R² scores to track their performance over time.

### Installation
#### Clone the repository:
```bash
git clone https://github.com/yerosan/Pharmaceuticals_sales_forecast.git
```
#### Install the required packages:
```bash
pip install -r requirements.txt
```
### Usage
Run the scripts in the notebook folder to preprocess data and build models. The app.py script can be used for running the main application to forcast future sales based on the input.

#### License
This project is licensed under the MIT License.
