import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups

def fetch_data():
    news = fetch_20newsgroups(subset = 'all', remove = ('headers', 'footers', 'quotes'))
    categories = news.target_names
    categories = [str(cat) for cat in categories]
    news_df = pd.DataFrame({'text':news.data, 'target':news.target})
    news_df['target_name'] = news_df['target'].apply(lambda x: categories[x])
    return news_df, np.array(categories)
    
def get_random_n_categories(n, df, categories, seed=3002):
    np.random.seed(seed)
    random_categories = np.random.choice(categories, n, replace = False)
    return random_categories

def get_subset_df(df, categories, selected_categories):
    df = df.copy()
    df = df[df['target_name'].isin(selected_categories)]
    df['target'] = df['target'].apply(lambda x: np.where(selected_categories == categories[x])[0][0])
    return df