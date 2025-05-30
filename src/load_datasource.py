# import Python libraries
import pandas as pd
import os

# Define to manage filenames
DATA_FILES = {
    "raw_analysit": "raw_analyst_ratings.csv",
    'AAPL':"AAPL_historical_data.csv",
    'AMZN':"AMZN_historical_data.csv",
    'GOOG':"GOOG_historical_data.csv",
    'META':"META_historical_data.csv",
    'MSFT':"MSFT_historical_data.csv",
    'NVDA':"NVDA_historical_data.csv",
    'TSLA':"TSLA_historical_data.csv"
}

# Get full path for 
def get_file_path(file_key):
    current_dir = os.getcwd()
    return os.path.join(current_dir, "../Data/raw_analyst_ratings.csv", DATA_FILES[file_key])

# get cleaned dataset
def get_cleaned_data(file_key):
    current_dir = os.getcwd()
    return os.path.join(current_dir, "../Data/raw_analyst_ratings.csv", DATA_FILES[file_key])

# Get full path for 
def get_stock_path(file_key):
    current_dir = os.getcwd()
    return os.path.join(current_dir, "../Data/yfinance_data/yfinance_data", DATA_FILES[file_key])


# Data loader class
class CSVData:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        return pd.read_csv(self.file_path)