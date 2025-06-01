import pandas as pd
import numpy as np
from pyti.simple_moving_average import simple_moving_average as sma
from pyti.relative_strength_index import relative_strength_index as rsi
from pyti.moving_average_convergence_divergence import moving_average_convergence_divergence as macd
from pyti.exponential_moving_average import exponential_moving_average as ema

import pynance as pn
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from matplotlib.ticker import PercentFormatter
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator


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

   
    def calculate_daily_returns(self):
        """Calculate daily returns (percentage change)."""
        self.df['Daily_Return'] = self.df['Close'].pct_change() * 100
        return self.df

    def plot_daily_returns(self):
        """Plot the calculated daily returns."""
        if 'Daily_Return' not in self.df.columns:
            self.calculate_daily_returns()
        
        plt.figure(figsize=(12, 5))
        
        # Create color list (red for negative, green for positive)
        colors = ['tomato' if x < 0 else 'limegreen' for x in self.df['Daily_Return']]
        
        # Plot bars
        plt.bar(self.df.index, self.df['Daily_Return'], color=colors, width=1, alpha=0.8)
        
        # Formatting
        plt.axhline(0, color='black', linewidth=0.5)
        plt.title('Daily Percentage Returns', fontsize=14, pad=20)
        plt.ylabel('Return (%)')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        
        # Date formatting
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

    def calculate_cumulative_returns(self):
        """Calculate cumulative returns over time."""
        if 'Close' not in self.df.columns:
            raise ValueError("DataFrame must contain 'Close' column")
        self.df['Cumulative_Return'] = (1 + self.df['Close'].pct_change()).cumprod() - 1
        return self.df
    
    def _ensure_datetime_index(self):
        """Convert index to datetime if needed"""
        if not isinstance(self.df.index, pd.DatetimeIndex):
            try:
                self.df.index = pd.to_datetime(self.df.index)
            except Exception as e:
                raise ValueError(f"Could not convert index to datetime: {e}")
    
    def _format_date_axis(self):
        """Smart date formatting based on timespan"""
        ax = plt.gca()
        try:
            days_span = (self.df.index[-1] - self.df.index[0]).days
            
            if days_span > 730:  # >2 years
                locator = mdates.YearLocator()
                fmt = mdates.DateFormatter('%Y')
            elif days_span > 180:  # 6m-2y
                locator = mdates.MonthLocator(interval=3)
                fmt = mdates.DateFormatter('%b %Y')
            else:  # <6 months
                locator = mdates.MonthLocator()
                fmt = mdates.DateFormatter('%d %b')
            
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(fmt)
            plt.xticks(rotation=45, ha='right')
        except Exception as e:
            print(f"Warning: Could not format dates - {e}")
    
    def plot_cumulative_returns(self):
        """Robust plotting with error handling"""
        try:
            if 'Cumulative_Return' not in self.df.columns:
                self.calculate_cumulative_returns()
            
            plt.figure(figsize=(12, 5))
            plt.plot(self.df.index, self.df['Cumulative_Return'], 
                    color='#6a0572', linewidth=2, alpha=0.8)
            
            plt.axhline(0, color='black', linewidth=0.7)
            plt.title('Cumulative Returns', fontsize=14, pad=20)
            plt.ylabel('Cumulative Return')
            plt.grid(True, linestyle=':', alpha=0.5)
            
            ax = plt.gca()
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            
            final_val = self.df['Cumulative_Return'].iloc[-1]
            plt.annotate(f'Final: {final_val:.1%}',
                        xy=(self.df.index[-1], final_val),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round', fc='w', alpha=0.8))
            
            self._format_date_axis()
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            plt.close()
            raise RuntimeError(f"Failed to plot cumulative returns: {e}")

    def annualized_volatility(self, trading_days=252):
        """Compute annualized volatility (standard deviation of daily returns)."""
        daily_volatility = self.df['Daily_Return'].std()
        annualized_vol = daily_volatility * np.sqrt(trading_days)
        return annualized_vol

    def sharpe_ratio(self, risk_free_rate=0.02, trading_days=252):
        """Compute Sharpe Ratio (risk-adjusted return)."""
        avg_daily_return = self.df['Daily_Return'].mean()
        daily_volatility = self.df['Daily_Return'].std()
        sharpe_ratio = (avg_daily_return - risk_free_rate) / daily_volatility * np.sqrt(trading_days)
        return sharpe_ratio

    def max_drawdown(self):
        """Calculate maximum drawdown (worst peak-to-trough decline)."""
        cumulative_returns = (1 + self.df['Close'].pct_change()).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min() * 100  # Convert to percentage
        return max_drawdown

    def beta(self, benchmark_returns, trading_days=252):
        """Compute Beta (market correlation). Requires benchmark returns (e.g., S&P 500)."""
        cov_matrix = np.cov(self.df['Daily_Return'].dropna(), benchmark_returns.dropna())
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        return beta

    def returns_enhanced(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Daily Returns with positive/negative colors
        colors = ['red' if x < 0 else 'green' for x in self.df['Daily_Return']]
        ax1.bar(self.df.index, self.df['Daily_Return'], color=colors, width=1)
        ax1.axhline(0, color='black', linewidth=0.5)
        ax1.set_title('Daily Returns (%)', fontsize=12, pad=20)
        
        # Cumulative Returns with annotations
        ax2.plot(self.df.index, self.df['Cumulative_Return'] * 100, 
                color='navy', linewidth=2.5)
        
        # Add final return annotation
        final_return = self.df['Cumulative_Return'].iloc[-1] * 100
        ax2.annotate(f'{final_return:.1f}%', 
                    xy=(self.df.index[-1], final_return),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        
        # Formatting
        for ax in [ax1, ax2]:
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.set_ylabel('Return (%)')
        ax2.yaxis.set_major_formatter(PercentFormatter())
        
        fig.autofmt_xdate(rotation=45)
        fig.suptitle('Enhanced Returns Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()
