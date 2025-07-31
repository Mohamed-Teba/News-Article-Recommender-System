import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import random
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt')
nltk.download('stopwords')

def load_data(file_path):
    col_news = ['NewsId', 'Category', 'SubCat', 'Title', 'Abstract', 'url', 'TitleEnt', 'AbstractEnt']
    try:
        df = pd.read_csv(file_path, sep='\t', header=None, names=col_news)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please provide the correct path to news.tsv.")
    
    df = df.sample(n=5000, random_state=42).reset_index(drop=True)
    
    df['Title'] = df['Title'].fillna('')
    df['Abstract'] = df['Abstract'].fillna('')
    
    df['Combined_Features'] = df['Title'] + ' ' + df['Abstract'] + ' ' + df['Category'] + ' ' + df['SubCat']
    
    return df

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stop_words]
    
    entity_keywords = {
        'trump', 'obama', 'biden', 'new york', 'california', 'london',
        'washington', 'china', 'russia', 'microsoft', 'apple', 'google',
        'facebook', 'amazon', 'tesla', 'united states', 'europe', 'asia'
    }
    tokens = [token for token in tokens if token.lower() not in entity_keywords]
    
    return ' '.join(tokens)

def extract_features(df):
    df['Processed_Features'] = df['Combined_Features'].apply(preprocess_text)
    
    tfidf = TfidfVectorizer(stop_words='english', min_df=0.001, max_df=0.999, ngram_range=(1, 3))
    feature_matrix = tfidf.fit_transform(df['Processed_Features'])
    
    return feature_matrix, tfidf, df

def create_user_profiles(df, num_users=100):
    user_profiles = {}
    for user_id in range(num_users):
        num_interactions = random.randint(5, 10)
        interacted_articles = df.sample(n=num_interactions, random_state=user_id)
        user_profiles[user_id] = {
            'interacted_articles': interacted_articles.index.tolist(),
            'categories': interacted_articles['Category'].tolist(),
            'subcategories': interacted_articles['SubCat'].tolist()
        }
    user_profiles[0]['interacted_articles'] = []
    user_profiles[0]['categories'] = []
    user_profiles[0]['subcategories'] = []
    return user_profiles

def compute_similarity(user_profile, feature_matrix):
    interacted_indices = user_profile['interacted_articles']
    if not interacted_indices:
        return np.zeros(feature_matrix.shape[0])
    
    max_index = feature_matrix.shape[0] - 1
    valid_indices = [idx for idx in interacted_indices if idx <= max_index]
    if not valid_indices:
        return np.zeros(feature_matrix.shape[0])
    
    user_vector = np.mean(feature_matrix[valid_indices].toarray(), axis=0).reshape(1, -1)
    similarities = cosine_similarity(user_vector, feature_matrix)
    return similarities[0]

def generate_recommendations(df, similarities, top_n=5, interacted_articles=None):
    if interacted_articles is None:
        interacted_articles = []
    recommendation_indices = np.argsort(similarities)[::-1]
    recommendation_indices = [idx for idx in recommendation_indices if idx not in interacted_articles][:top_n]
    
    recommendations = df.iloc[recommendation_indices][['Title', 'Abstract', 'Category', 'SubCat']]
    recommendations['Similarity_Score'] = similarities[recommendation_indices]
    return recommendations

def recommend_popular_items(df, top_n=5):
    popular_categories = df['Category'].value_counts().head(3).index
    popular_articles = df[df['Category'].isin(popular_categories)].sample(n=top_n, random_state=42)
    return popular_articles[['Title', 'Abstract', 'Category', 'SubCat']]

def evaluate_recommendations(recommendations, relevant_articles, top_n=5):
    recommended_titles = set(recommendations['Title'].head(top_n))
    relevant_titles = set(relevant_articles['Title'])
    
    true_positives = len(recommended_titles & relevant_titles)
    precision = true_positives / top_n if top_n > 0 else 0
    recall = true_positives / len(relevant_titles) if len(relevant_titles) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score, true_positives

def run_streamlit_app(df, feature_matrix, user_profiles):
    st.set_page_config(page_title="News Recommender", layout="wide")
    st.title("📰 News Article Recommender System")
    
    st.sidebar.header("User Preferences")
    user_id = st.sidebar.selectbox("Select User ID", list(user_profiles.keys()), key="user_id")
    top_n = st.sidebar.slider("Number of Recommendations", 1, 10, 5, key="top_n")
    
    st.header(f"Recommendations for User {user_id}")
    
    user_profile = user_profiles[user_id]
    
    if not user_profile['interacted_articles']:
        st.subheader("New User Detected")
        st.write("Since you have no interaction history, here are some popular articles:")
        recommendations = recommend_popular_items(df, top_n)
    else:
        similarities = compute_similarity(user_profile, feature_matrix)
        recommendations = generate_recommendations(
            df, similarities, top_n, user_profile['interacted_articles']
        )
    
    st.subheader("Recommended Articles")
    for idx, row in recommendations.iterrows():
        with st.container():
            st.markdown(f"**Title**: {row['Title']}")
            st.markdown(f"**Abstract**: {row['Abstract']}")
            st.markdown(f"**Category**: {row['Category']}")
            st.markdown(f"**Subcategory**: {row['SubCat']}")
            if 'Similarity_Score' in row:
                st.markdown(f"**Similarity Score**: {row['Similarity_Score']:.2f}")
            st.markdown("---")
    
    st.subheader("Evaluate Recommendations")
    if st.button("Calculate Metrics"):
        relevant_articles = df[df['Category'].isin(user_profile['categories'])]
        precision, recall, f1_score, true_positives = evaluate_recommendations(
            recommendations, relevant_articles, top_n
        )
        st.markdown(f"**Precision**: {precision:.2f}")
        st.markdown(f"**Recall**: {recall:.2f}")
        st.markdown(f"**F1-Score**: {f1_score:.2f}")
        st.markdown(f"**True Positives**: {true_positives}")

if __name__ == "__main__":
    file_path = "C:/Users/WDT/Desktop/Content_Based_Filtering_Recommender_System/news.tsv"
    try:
        df = load_data(file_path)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    
    feature_matrix, tfidf, df = extract_features(df)
    
    user_profiles = create_user_profiles(df)
    
    run_streamlit_app(df, feature_matrix, user_profiles)