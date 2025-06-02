import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
from mplfinance.original_flavor import candlestick_ohlc
import pandas as pd

class StockVisualizer:
    
    def __init__(self, path):
        """Initialize with data validation and preprocessing"""
        self.df = path.copy()
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Ensure proper datetime index and required columns"""
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index)
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def plot_price_volume(self, figsize=(14, 8)):
        """Plot price and volume with professional formatting"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                    gridspec_kw={'height_ratios': [3, 1]})
        
        # Prepare OHLC data using the datetime index
        ohlc = self.df[['Open', 'High', 'Low', 'Close']].copy()
        ohlc.reset_index(inplace=True)
        
        # Convert datetime to numerical format for candlestick plot
        ohlc['Date_num'] = ohlc[self.df.index.name or 'index'].map(mdates.date2num)
        
        # Create list of tuples in correct format for candlestick_ohlc
        ohlc_values = ohlc[['Date_num', 'Open', 'High', 'Low', 'Close']].values
        
        # Candlestick plot
        candlestick_ohlc(ax1, ohlc_values, width=0.6, 
                        colorup='g', colordown='r', alpha=0.8)
        ax1.set_title('Price and Volume Analysis', fontsize=16, pad=20)
        ax1.set_ylabel('Price ($)')
        ax1.grid(True, linestyle=':', alpha=0.5)
        
        # Volume plot
        ax2.bar(self.df.index, self.df['Volume'], 
            color=self._volume_colors(), width=1, alpha=0.7)
        ax2.set_ylabel('Volume')
        ax2.grid(True, linestyle=':', alpha=0.5)
        
        self._format_date_axis([ax1, ax2])
        plt.tight_layout()
        plt.show()
        
    def plot_technical_indicators(self, figsize=(14, 10)):
        """Plot multiple technical indicators"""
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        # Price with moving averages
        axes[0].plot(self.df.index, self.df['Close'], label='Close', color='navy')
        if 'SMA_50' in self.df.columns:
            axes[0].plot(self.df.index, self.df['SMA_50'], label='50-day SMA', color='orange')
        if 'EMA_20' in self.df.columns:
            axes[0].plot(self.df.index, self.df['EMA_20'], label='20-day EMA', color='purple')
        axes[0].set_title('Technical Indicators', fontsize=16, pad=20)
        axes[0].legend()
        axes[0].grid(True, linestyle=':', alpha=0.5)
        
        # RSI
        if 'RSI_14' in self.df.columns:
            axes[1].plot(self.df.index, self.df['RSI_14'], label='RSI (14)', color='teal')
            axes[1].axhline(70, color='red', linestyle='--')
            axes[1].axhline(30, color='green', linestyle='--')
            axes[1].set_ylim(0, 100)
            axes[1].legend()
            axes[1].grid(True, linestyle=':', alpha=0.5)
        
        # MACD
        if 'MACD' in self.df.columns:
            axes[2].plot(self.df.index, self.df['MACD'], label='MACD', color='blue')
            axes[2].plot(self.df.index, self.df['MACD_Signal'], label='Signal', color='orange')
            axes[2].bar(self.df.index, self.df['MACD_Hist'], 
                       color=self._macd_hist_colors(), label='Histogram')
            axes[2].legend()
            axes[2].grid(True, linestyle=':', alpha=0.5)
        
        # Bollinger Bands
        if 'BB_Upper' in self.df.columns:
            axes[3].plot(self.df.index, self.df['Close'], label='Price', color='navy')
            axes[3].plot(self.df.index, self.df['BB_Upper'], label='Upper Band', color='red')
            axes[3].plot(self.df.index, self.df['BB_Middle'], label='Middle Band', color='gray')
            axes[3].plot(self.df.index, self.df['BB_Lower'], label='Lower Band', color='green')
            axes[3].legend()
            axes[3].grid(True, linestyle=':', alpha=0.5)
        
        self._format_date_axis(axes)
        plt.tight_layout()
        plt.show()
    
    def _volume_colors(self):
        """Generate colors based on price movement"""
        return ['green' if close >= open_ else 'red' 
               for close, open_ in zip(self.df['Close'], self.df['Open'])]
    
    def _macd_hist_colors(self):
        """Generate MACD histogram colors"""
        return ['green' if x >= 0 else 'red' for x in self.df['MACD_Hist']]
    
    def _format_date_axis(self, axes):
        """Smart date formatting for all axes"""
        days_span = (self.df.index[-1] - self.df.index[0]).days
        
        for ax in axes:
            if days_span > 730:  # >2 years
                locator = mdates.YearLocator()
                fmt = mdates.DateFormatter('%Y')
            elif days_span > 180:  # 6m-2y
                locator = mdates.MonthLocator(interval=3)
                fmt = mdates.DateFormatter('%b %Y')
            else:  # <6 months
                locator = mdates.WeekdayLocator()
                fmt = mdates.DateFormatter('%d %b')
            
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(fmt)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')