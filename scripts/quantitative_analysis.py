import pandas as pd
from pyti.simple_moving_average import simple_moving_average as sma
from pyti.relative_strength_index import relative_strength_index as rsi
from pyti.moving_average_convergence_divergence import moving_average_convergence_divergence as macd
from pyti.exponential_moving_average import exponential_moving_average as ema

import pynance as pn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import spearmanr


class StockAnalysis:
    def __init__(self, df):
        # Ensure df is a DataFrame
        if isinstance(df, pd.DataFrame):
            self.df = df
        else:
            raise ValueError("Expected a pandas DataFrame")

    def calculate_indicators(self, period=14):
        # Ensure data is sorted
        self.df = self.df.sort_values(by="Date")

        # SMA and RSI Calculation
        self.df["SMA_20"] = sma(self.df["Close"].tolist(), period=50)
        self.df["RSI"] = rsi(self.df["Close"].tolist(), period=period)

        # MACD Calculation
        macd_values = macd(self.df["Close"].tolist(), short_period=12, long_period=26)
        self.df["MACD"] = macd_values  # MACD line (directly returned by pyti's macd())

        # Signal Line Calculation (SMA of MACD values)
        signal_line = sma(macd_values, period=9)
        self.df["MACD_signal"] = signal_line

        # Histogram Calculation (MACD - Signal Line)
        self.df["MACD_hist"] = self.df["MACD"] - self.df["MACD_signal"]

        self.df["Close"] = pd.to_numeric(self.df["Close"], errors='coerce')
        self.df.dropna(subset=["Close"], inplace=True)

    def visualize_SMA(self):
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.df["Date"], self.df["Close"], label="Close Price", color="blue")
        plt.plot(self.df["Date"], self.df["SMA_20"], label="SMA_20", color="orange")
        plt.legend()
        plt.show()

    def visualize_RSI(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.df["Date"], self.df["RSI"], label="RSI", color="green")
        plt.axhline(70, linestyle="--", color="red", label="Overbought")
        plt.axhline(30, linestyle="--", color="blue", label="Oversold")
        plt.legend()
        plt.show()

    def visualize_MACD(self):
        #Plot MACD, signal line, and histogram to identify bullish/bearish momentum.
        plt.figure(figsize=(12, 6))
        plt.plot(self.df["Date"], self.df["MACD"], label="MACD", color="purple")
        plt.plot(self.df["Date"], self.df["MACD_signal"], label="Signal Line", color="orange")
        plt.bar(self.df["Date"], self.df["MACD_hist"], label="Histogram", color="gray")
        plt.legend()
        plt.show()


