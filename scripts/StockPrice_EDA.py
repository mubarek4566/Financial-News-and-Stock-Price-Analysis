import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import talib
import math
from matplotlib.ticker import PercentFormatter


class Stock_EDA:
    def __init__(self, path):
        self.df = path

    def plot_distribution(self):
        numerical_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        num_plots = len(numerical_cols)
        
        # Create 2x3 subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for i, col in enumerate(numerical_cols):
            sns.histplot(self.df[col], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}', fontsize=12)
        
        # Hide any unused axes (if fewer than total grid cells)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()


    def univariate_num(self):
        """
        Univariate Analysis for Numerical Columns Distribution.
        """
        # Select only numerical columns
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        # Define grid size: Calculate rows and columns dynamically
        num_cols = 3  # Number of columns in the grid
        num_plots = len(numerical_cols)  # Total number of plots
        num_rows = math.ceil(num_plots / num_cols)  # Calculate required rows dynamically
        
        # Create a figure with subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        axes = axes.flatten()  # Flatten axes for easier iteration
        
        # Loop through numerical columns and plot histograms
        for idx, col in enumerate(numerical_cols):
            sns.histplot(self.df[col], kde=True, bins=30, ax=axes[idx])  # Histogram with KDE
            axes[idx].set_title(f"Distribution of {col}")  # Set title
            axes[idx].set_xlabel(col)  # Label x-axis
            axes[idx].set_ylabel("Frequency")  # Label y-axis
        
        # Hide any unused subplots
        for i in range(len(numerical_cols), len(axes)):
            fig.delaxes(axes[i])
        
        # Adjust layout to prevent overlapping
        plt.tight_layout()
        plt.show()

    def bivariate_num(self):
        # Bivariate Analysis for Numerical vs Numerical (Scatterplots)
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        num_plots = len(numerical_cols) - 1  # Exclude the first column as x-axis
        num_cols = 3  # Number of columns in the grid
        num_rows = math.ceil(num_plots / num_cols)  # Calculate rows needed for grid
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))  # Flexible grid size
        axes = axes.flatten()  # Flatten in case of multi-dimensional axes
        
        x_col = numerical_cols[0]  # Use the first numerical column as x-axis
        plot_idx = 0  # Track plot index

        for col in numerical_cols:
            if col != x_col:  # Skip the x-axis column itself
                ax = axes[plot_idx]
                sns.scatterplot(data=self.df, x=x_col, y=col, ax=ax)
                ax.set_title(f"{x_col} vs {col}")
                ax.set_xlabel(x_col)
                ax.set_ylabel(col)
                plot_idx += 1
        
        # Hide any unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()

    def plot_returns_enhanced(self):
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