import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Stock_EDA:
    def __init__(self, path):
        self.df = path

    def plot_distribution(self):
        numerical_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in numerical_cols:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.df[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.show()