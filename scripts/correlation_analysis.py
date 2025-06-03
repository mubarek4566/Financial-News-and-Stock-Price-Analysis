import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from scipy.stats import pearsonr
from scipy.stats import spearmanr

nltk.download('vader_lexicon')  # Run this only once

# Step 2: Initialize analyzer
sid = SentimentIntensityAnalyzer()

class CorrelationAnalysis:
    def __init__(self, path):
        self.df = path
        
    # Perform sentiment analysis
    def get_sentiment(self, text):
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'

    def compute_daily_returns(self, date_col='date', price_col='Close'):
        """
        Computes daily return from stock price data.
        Returns:
            pd.DataFrame: DataFrame with 'daily_return' and 'daily_return_percent' columns.
        """
        #df = stock_df.copy()
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df = self.df.sort_values(date_col)
        self.df['daily_return'] = self.df[price_col].pct_change()
        self.df['daily_return_percent'] = self.df['daily_return'] * 100
        return self.df
    
    def plot_category(self):
        # Plot the categories
        plt.figure(figsize=(8, 5))
        sns.barplot(x=self.df.index, y=self.df.values, palette="coolwarm")
        plt.title("Distribution of Daily Returns")
        plt.xlabel("Return Category")
        plt.ylabel("Count")
        plt.show()
    
    def align_returns_with_sentiment(sentiment_df, stock_returns_df, sentiment_date_col='aligned_date', stock_date_col='Date'):
        """
        Aligns stock return data with sentiment data based on date.

        Parameters:
            sentiment_df (pd.DataFrame): News sentiment data with aligned dates.
            stock_returns_df (pd.DataFrame): Stock data with daily returns.
            sentiment_date_col (str): Column in sentiment_df to match on.
            stock_date_col (str): Column in stock_returns_df to match on.

        Returns:
            pd.DataFrame: Merged DataFrame with sentiment and stock return info.
        """
        merged_df = sentiment_df.merge(
            stock_returns_df[[stock_date_col, 'daily_return', 'daily_return_percent']],
            left_on=sentiment_date_col,
            right_on=stock_date_col,
            how='left'
        )
        return merged_df

    def Spearman_Corr_news_stock(self):
            required_columns = ['sentiment', 'daily_return']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                raise KeyError(f"The following required columns are missing: {missing_columns}")
            
            # Map sentiments to numerical scores (if not already done)
            sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
            if 'sentiment_score' not in self.df.columns:
                self.df['sentiment_score'] = self.df['sentiment'].map(sentiment_map)
            
            # Drop rows with NaN in the necessary columns
            self.df.dropna(subset=['sentiment_score', 'daily_return'], inplace=True)
            
            # Compute Spearman correlation
            correlation, p_value = spearmanr(self.df['sentiment_score'], self.df['daily_return'])
            print(f"Spearman Correlation: {correlation:.4f} (p-value: {p_value:.4e})")
            
            # Scatterplot
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=self.df, x='sentiment_score', y='daily_return', color="blue")
            plt.title(f"Correlation Between Sentiment Scores and Stock Returns\n(Spearman ρ: {correlation:.2f})")
            plt.xlabel("Sentiment Score")
            plt.ylabel("Daily Stock Return (%)")
            plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
            plt.show()

            