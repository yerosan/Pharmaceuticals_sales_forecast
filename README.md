# Pharmaceuticals Sales Forecast

This project aims to predict sales for pharmaceutical stores using time-series data, considering various factors like promotions, store type, and holiday impacts. The analysis helps businesses optimize promotional strategies and inventory management to improve revenue and operational efficiency.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [License](#license)

## Project Overview

This project focuses on:
- **Exploratory Data Analysis (EDA)**: Insights on how promotions, store types, and holidays affect sales.
- **Sales Forecasting**: Predicting future sales using various machine learning models.
- **Optimization**: Recommending strategies based on the analysis to improve sales performance.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yerosan/Pharmaceuticals_sales_forecast.git
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

Pharmaceuticals_sales_forecast/
```bash
│ ├── data/ # Datasets for analysis 
│         ├── train.csv # Training data 
│         ├── test.csv # Test data 
│ ├── notebooks/ # Jupyter notebooks for analysis 
│              ├── exploratory_data_analysis.ipynb # Exploratory Data Analysis notebook 
│              ├── Model_Building.ipynb # Sales prediction models notebook 
│ ├── scripts/ # Python scripts for reproducibility 
│            ├── data_preprocessing.py # Data cleaning and preprocessing functions 
│            ├── data_accessing.py # Reading data from the provided directory
│ ├── README.md # Project documentation 
└── requirements.txt # Required Python packages 

```

## Usage

- **Run Exploratory Data Analysis**: Open the `exploratory_data_analysis.ipynb` notebook in the `notebooks/` folder and execute the cells to explore the sales data.


- **Data Processing**: Run the scripts in the `scripts/` folder to preprocess data:
    ```bash
    python scripts/data_preprocessing.py
    ```

## Results

The analysis revealed:
- Promotions significantly increase sales, especially for certain store types.
- Holidays such as Christmas show high sales peaks.
- Store type B showed the highest overall sales during promotional periods.

## License

This project is licensed under the MIT License. See the LICENSE file for details.


