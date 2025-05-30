
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import zscore

from nltk.corpus import stopwords

# Download NLTK stopwords if not already done
nltk.download('stopwords')


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
        
    def publication_trends(self, column ='date',resample_freq='D'):
        # Analyze the trends of publication by date
        self.df[column] = pd.to_datetime(self.df[column], errors='coerce')

        # Drop rows with invalid dates
        self.df = self.df.dropna(subset=[column])

        # Group and count publications by date
        pub_trends = self.df.set_index(column).resample(resample_freq).size()
        print(pub_trends)
        
        # Plot
        plt.figure(figsize=(14, 6))
        sns.lineplot(x=pub_trends.index, y=pub_trends.values, color='steelblue')
        plt.title(f"News Publication Trend Over Time ({resample_freq} Frequency)")
        plt.xlabel("Date")
        plt.ylabel("Number of Articles")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def extract_keywords(self, column='headline', ngram_range=(1, 2), top_n=20):
        # Preprocess text
        stop_words = set(stopwords.words('english'))

        def clean_text(text):
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            tokens = text.split()
            tokens = [t for t in tokens if t not in stop_words]
            return ' '.join(tokens)

        cleaned_headlines = self.df[column].dropna().apply(clean_text)

        # Vectorize
        vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
        X = vectorizer.fit_transform(cleaned_headlines)

        # Sum and sort
        sum_words = X.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_n]

        # Create DataFrame for plotting
        keywords_df = pd.DataFrame(words_freq, columns=['phrase', 'count'])
        print(keywords_df)
    
        # Plot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=keywords_df, x='count', y='phrase', hue='phrase', palette='crest', legend=False)
        plt.title(f"Top {top_n} Keywords/Phrases in Headlines")
        plt.xlabel("Frequency")
        plt.ylabel("Keyword/Phrase")
        plt.tight_layout()
        plt.show()

    from scipy.stats import zscore

    def publication_spikes(self, column='date', threshold_zscore=2.0):
        # Analyze and visualize spikes in publication frequency over time.

        # Convert to datetime
        self.df[column] = pd.to_datetime(self.df[column], errors='coerce')
        self.df = self.df.dropna(subset=[column])

        # Count articles per day
        daily_counts = self.df.groupby(self.df[column].dt.date).size()
        daily_counts.index = pd.to_datetime(daily_counts.index)
        daily_counts = daily_counts.sort_index()


    def process_dates(self: pd.DataFrame, column: str = 'date') -> pd.DataFrame:
        # Convert to datetime # Convert to datetime
        self.df['date'] = pd.to_datetime(self.df['date'], format='mixed', errors='coerce')

        # Extract year, month, day
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['day'] = self.df['date'].dt.day
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        #return self.df

    def plot_articles_by_day_of_week(self: pd.DataFrame):
        """Plot number of articles per day of the week."""
        day_counts = self.df['day_of_week'].value_counts().sort_index()
        day_counts.plot(kind='bar', title='Articles by Day of the Week')
        plt.xlabel('Day')
        plt.ylabel('Number of Articles')
        plt.tight_layout()
        plt.show()

    def plot_weekly_article_trends(self: pd.DataFrame, column: str = 'date'):
        """Plot number of articles published per week."""
        self.df.set_index('date').resample('W').size().plot(title='Weekly Article Frequency')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.tight_layout()
        plt.show()

    def identify_publication_spikes(self, date_column='date', threshold_zscore=2.0):
        # Convert to datetime
        self.df[date_column] = pd.to_datetime(self.df[date_column], errors='coerce')
        self.df = self.df.dropna(subset=[date_column])

        # Count articles per day
        daily_counts = self.df.groupby(self.df[date_column].dt.date).size()
        daily_counts.index = pd.to_datetime(daily_counts.index)
        daily_counts = daily_counts.sort_index()

        # Calculate z-score for spike detection
        zscores = zscore(daily_counts)
        spike_mask = zscores > threshold_zscore

        # Combine data
        result = pd.DataFrame({
            'article_count': daily_counts,
            'z_score': zscores,
            'is_spike': spike_mask
        })
        print(result)

        # Plot
        plt.figure(figsize=(15, 6))
        sns.lineplot(x=daily_counts.index, y=daily_counts.values, label='Daily Article Count', color='steelblue')
        plt.scatter(daily_counts.index[spike_mask], daily_counts[spike_mask], color='red', label='Spikes', zorder=5)
        plt.axhline(daily_counts.mean(), linestyle='--', color='gray', label='Average')
        plt.title("Publication Frequency Over Time with Spikes")
        plt.xlabel("Date")
        plt.ylabel("Number of Articles")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def most_active_publishers(self, publisher_col='publisher', top_n=10):
        
        # Returns the top N publishers by article count.
        top_publisher = self.df[publisher_col].value_counts().head(top_n)
        print(top_publisher)

        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(y=top_publisher.index, x=top_publisher.values, palette="mako")
        plt.title(f"Top {top_n} Most Active Publishers")
        plt.xlabel("Number of Articles")
        plt.ylabel("Publisher")
        plt.tight_layout()
        plt.show()

    def top_keywords_by_publisher(self, publisher_col='publisher', text_col='headline', publisher='Paul Quintaro', top_n=10):

        #Get top keywords for a specific publisher.

        stop_words = set(stopwords.words('english'))

        def clean_text(text):
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            tokens = text.split()
            tokens = [t for t in tokens if t not in stop_words]
            return ' '.join(tokens)

        # Filter and clean
        pub_df = self.df[self.df[publisher_col] == publisher]
        clean_headlines = pub_df[text_col].dropna().apply(clean_text)

        # Count keywords
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(clean_headlines)
        sum_words = X.sum(axis=0)
        keywords = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        keywords = sorted(keywords, key=lambda x: x[1], reverse=True)[:top_n]

        # Plot
        keywords_df = pd.DataFrame(keywords, columns=['Keyword', 'Count'])
        plt.figure(figsize=(10, 5))
        sns.barplot(data=keywords_df, x='Count', y='Keyword', color='steelblue')
        plt.title(f"Top {top_n} Keywords Reported by {publisher}")
        plt.tight_layout()
        plt.show()

        return keywords_df
    
    def analyze_email_domains(self, publisher_col='publisher', top_n=10):
       # Extracts domains from email addresses in the publisher column and visualizes the most frequent ones.
        
        # Define regex pattern to match email addresses
        email_pattern = r'[\w\.-]+@([\w\.-]+\.\w+)'

        # Extract domains
        self.df['email_domain'] = self.df[publisher_col].str.extract(email_pattern)

        # Drop NaNs (non-email publishers)
        domain_counts = self.df['email_domain'].dropna().value_counts().head(top_n)

        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=domain_counts.values, y=domain_counts.index, legend=False)
        plt.title(f"Top {top_n} Publisher Domains from Emails")
        plt.xlabel("Number of Articles")
        plt.ylabel("Email Domain")
        plt.tight_layout()
        plt.show()

        return domain_counts

