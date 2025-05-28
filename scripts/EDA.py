
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    def __init__(self, dataframe):
        #
        self.df = dataframe

    def data_info(self):
        print('----' + 'Dataset Info.' + '-------')
        print(self.df.info())
        print('\n\n ----' + 'Identify Null values.' + '-------')
        print(self.df.isnull().sum())
        print('\n----' + 'Shape of the dataset.' + '-------')
        print(self.df.shape)

    def stat_summary(self):
        print("\n Numerical Features Summary:")
        print(self.df.describe().T)
        print("\n Categorical Features Summary:")
        print(self.df.describe(include='object')) 

    def descriptive_stat(self):
        print('Textual Length of Headline')
        self.df['headline_length'] = self.df['headline'].apply(len)
        print(self. df['headline_length'].describe())

        print("\n The Longest Headlines:")
        print(self.df.sort_values(by='headline_length', ascending=False)[['headline', 'headline_length']].head())

        # Plot histogram of headline lengths
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['headline_length'], bins=15, kde=True, color='skyblue')
        plt.title("Distribution of Headline Lengths")
        plt.xlabel("Number of Characters in Headline")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def article_per_publisher(self, column='publisher', plot=True, top_n=10):
        print('\nTop-10 Articles Publisher')
        publisher_count = self.df['publisher'].value_counts()
        print(publisher_count.head(top_n))

        if plot:
            plt.figure(figsize=(12, 6))
            sns.barplot(
                x=publisher_count.head(top_n).values,
                y=publisher_count.head(top_n).index,
                hue=publisher_count.head(top_n).index,
                legend=False,
                palette="viridis"
            )
            plt.title(f"Top {top_n} Most Active Publishers")
            plt.xlabel("Number of Articles")
            plt.ylabel("Publisher")
            plt.tight_layout()
            plt.show()
        
    
            