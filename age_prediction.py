import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_age_examples(csv_path):
    df = pd.read_csv(csv_path)
    return df

def preprocess_examples(df):
    vectorizer = TfidfVectorizer().fit_transform(df['Examples'])
    return vectorizer

def compute_similarity(user_input, vectorizer, examples):
    user_vector = TfidfVectorizer().fit_transform([user_input])
    similarities = cosine_similarity(user_vector, vectorizer)
    return similarities
