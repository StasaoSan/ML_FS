import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(df):
    custom_mapping = {'Neutral': 0, 'Positive': 1, 'Negative': 2,
                      'Extremely Positive': 3, 'Extremely Negative': 4}
    df['Sentiment'] = df['Sentiment'].map(custom_mapping)

    y = df['Sentiment']
    X_text = df['OriginalTweet']

    X_location = df['Location'].fillna('Unknown')
    X_date = pd.to_datetime(df['TweetAt'], errors='coerce', dayfirst=True)

    df['DayOfWeek'] = X_date.dt.dayofweek
    df['IsWeekend'] = (X_date.dt.dayofweek >= 5).astype(int)

    vectorizer = TfidfVectorizer()
    X_text_vectorized = vectorizer.fit_transform(X_text).toarray()

    encoder = OneHotEncoder(handle_unknown='ignore')
    X_location_encoded = encoder.fit_transform(X_location.values.reshape(-1, 1)).toarray()

    X_time = df[['DayOfWeek', 'IsWeekend']].values

    X = np.hstack((X_text_vectorized, X_location_encoded, X_time))
    print(f"X shape: {X.shape}")

    text_features = vectorizer.get_feature_names_out()
    location_features = encoder.get_feature_names_out(['Location'])
    time_features = ['DayOfWeek', 'IsWeekend']

    all_feature_names = np.array(list(text_features) + list(location_features) + time_features, dtype=object)

    return X, y, all_feature_names


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def get_processed_data(file_path='Corona_NLP_test.csv'):
    df = load_data(file_path)
    X, y, all_feature_names = preprocess_data(df)

    X_train, X_val, y_train, y_val = split_data(X, y)

    return X_train, y_train.values, X_val, y_val.values, all_feature_names
