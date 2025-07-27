"""
Sentiment-Based Product Recommendation System - Model Module

This module contains the core machine learning models and recommendation logic
for the Flask web application deployment.

Author: Capstone Project
Date: 2025
"""

import pickle
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Custom unpickler to handle class name mapping
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Map __main__.ItemBasedCF to model.ItemBasedCF
        if module == '__main__' and name == 'ItemBasedCF':
            module = 'model'
        return super().find_class(module, name)

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)


# Collaborative Filtering Classes
class ItemBasedCF:
    def __init__(self, rating_matrix, k=50):
        self.rating_matrix = rating_matrix
        self.k = k  # Number of similar items to consider
        self.item_similarity = None

    def compute_item_similarity(self):
        """Compute item-item similarity matrix using cosine similarity"""
        print("Computing item-item similarity matrix...")

        # Transpose the matrix to compute item-item similarity
        item_matrix = self.rating_matrix.T

        # Compute cosine similarity
        self.item_similarity = cosine_similarity(item_matrix)

        # Set diagonal to 0 (item shouldn't be similar to itself for recommendations)
        np.fill_diagonal(self.item_similarity, 0)

        print(f"Item similarity matrix shape: {self.item_similarity.shape}")

    def predict_rating(self, user_idx, item_idx):
        """Predict rating for a user-item pair"""
        if self.item_similarity is None:
            self.compute_item_similarity()

        # Find items rated by this user
        items_rated_by_user = np.where(self.rating_matrix[user_idx, :] > 0)[0]

        if len(items_rated_by_user) == 0:
            # User hasn't rated any items, return global average
            return np.mean(self.rating_matrix[self.rating_matrix > 0])

        # Get similarities with items rated by this user
        similarities = self.item_similarity[item_idx, items_rated_by_user]
        ratings = self.rating_matrix[user_idx, items_rated_by_user]

        # Select top-k similar items
        if len(similarities) > self.k:
            top_k_indices = np.argsort(similarities)[-self.k:]
            similarities = similarities[top_k_indices]
            ratings = ratings[top_k_indices]

        # Remove items with zero similarity
        non_zero_mask = similarities > 0
        if not np.any(non_zero_mask):
            return np.mean(self.rating_matrix[self.rating_matrix > 0])

        similarities = similarities[non_zero_mask]
        ratings = ratings[non_zero_mask]

        # Weighted average prediction
        if np.sum(similarities) == 0:
            return np.mean(ratings)

        predicted_rating = np.sum(similarities * ratings) / np.sum(similarities)
        return predicted_rating

    def recommend_items(self, user_idx, n_recommendations=10):
        """Recommend top N items for a user"""
        if self.item_similarity is None:
            self.compute_item_similarity()

        # Get items not rated by the user
        user_ratings = self.rating_matrix[user_idx, :]
        unrated_items = np.where(user_ratings == 0)[0]

        if len(unrated_items) == 0:
            return []

        # Predict ratings for unrated items
        predictions = []
        for item_idx in unrated_items:
            pred_rating = self.predict_rating(user_idx, item_idx)
            predictions.append((item_idx, pred_rating))

        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]


class SentimentBasedRecommendationSystem:
    """
    Main class for the sentiment-based product recommendation system
    """
    
    def __init__(self):
        """Initialize the recommendation system"""
        self.sentiment_model = None
        self.vectorizer = None
        self.label_encoder = None
        self.cf_system = None
        self.user_mappings = None
        self.item_mappings = None
        self.rating_matrix = None
        self.processed_data = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_models(self):
        """Load all trained models and data"""
        try:
            # Load sentiment analysis components
            with open('final_sentiment_model.pkl', 'rb') as f:
                self.sentiment_model = pickle.load(f)
            
            # Load vectorizer (try TF-IDF first, then Count)
            try:
                with open('tfidf_vectorizer.pkl', 'rb') as f:
                    self.vectorizer = pickle.load(f)
            except FileNotFoundError:
                with open('count_vectorizer.pkl', 'rb') as f:
                    self.vectorizer = pickle.load(f)
            
            with open('label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Load recommendation system components using custom unpickler
            with open('final_recommendation_system.pkl', 'rb') as f:
                self.cf_system = CustomUnpickler(f).load()
            
            with open('user_mappings.pkl', 'rb') as f:
                self.user_mappings = pickle.load(f)
            
            with open('item_mappings.pkl', 'rb') as f:
                self.item_mappings = pickle.load(f)
            
            self.rating_matrix = np.load('rating_matrix.npy')
            
            # Load processed data
            self.processed_data = pd.read_csv('preprocessed_data.csv')
            
            print("All models and data loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    
    def preprocess_text(self, text, use_lemmatization=True):
        """
        Preprocess text for sentiment analysis
        
        Args:
            text (str): Input text to preprocess
            use_lemmatization (bool): Whether to use lemmatization
        
        Returns:
            str: Preprocessed text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags, URLs, emails
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        # Apply lemmatization
        if use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(word, self.get_wordnet_pos(word)) for word in tokens]
        
        return ' '.join(tokens)
    
    def predict_sentiment(self, text):
        """
        Predict sentiment for a given text
        
        Args:
            text (str): Text to analyze
        
        Returns:
            tuple: (sentiment_label, confidence_score)
        """
        if not text or pd.isna(text):
            return 'Neutral', 0.5
        
        try:
            # Preprocess the text
            processed_text = self.preprocess_text(text, use_lemmatization=True)
            
            if not processed_text:
                return 'Neutral', 0.5
            
            # Vectorize the text
            text_vector = self.vectorizer.transform([processed_text])
            
            # Predict sentiment
            prediction = self.sentiment_model.predict(text_vector)[0]
            prediction_proba = self.sentiment_model.predict_proba(text_vector)[0]
            
            # Convert prediction to label
            sentiment_label = self.label_encoder.inverse_transform([prediction])[0]
            confidence = max(prediction_proba)
            
            return sentiment_label, confidence
        
        except Exception as e:
            print(f"Error predicting sentiment: {e}")
            return 'Neutral', 0.5
    
    def calculate_product_sentiment_score(self, product_name):
        """
        Calculate sentiment score for a product based on its reviews
        
        Args:
            product_name (str): Name of the product
        
        Returns:
            dict: Dictionary containing sentiment analysis results
        """
        # Get all reviews for this product
        product_reviews = self.processed_data[self.processed_data['name'] == product_name]
        
        if len(product_reviews) == 0:
            return {
                'product_name': product_name,
                'total_reviews': 0,
                'positive_count': 0,
                'negative_count': 0,
                'positive_percentage': 0.0,
                'average_rating': 0.0,
                'sentiment_score': 0.0
            }
        
        # Use existing sentiment labels
        sentiments = product_reviews['user_sentiment'].values
        positive_count = sum(sentiments == 'Positive')
        negative_count = sum(sentiments == 'Negative')
        
        total_reviews = len(product_reviews)
        positive_percentage = (positive_count / total_reviews) * 100 if total_reviews > 0 else 0
        average_rating = product_reviews['reviews_rating'].mean()
        
        # Calculate combined sentiment score
        normalized_rating = ((average_rating - 1) / 4) * 100
        sentiment_score = (positive_percentage * 0.7) + (normalized_rating * 0.3)
        
        return {
            'product_name': product_name,
            'total_reviews': total_reviews,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'positive_percentage': round(positive_percentage, 2),
            'average_rating': round(average_rating, 2),
            'sentiment_score': round(sentiment_score, 2)
        }
    
    def get_user_recommendations(self, username, n_recommendations=20):
        """
        Get collaborative filtering recommendations for a user
        
        Args:
            username (str): Username to get recommendations for
            n_recommendations (int): Number of recommendations
        
        Returns:
            list: List of tuples (item_name, predicted_rating)
        """
        if username not in self.user_mappings['user_to_idx']:
            return []
        
        user_idx = self.user_mappings['user_to_idx'][username]
        
        # Get recommendations from the CF system
        recommendations = self.cf_system.recommend_items(user_idx, n_recommendations)
        
        # Convert item indices to item names
        recommendations_with_names = []
        for item_idx, predicted_rating in recommendations:
            item_name = self.item_mappings['idx_to_item'][item_idx]
            recommendations_with_names.append((item_name, predicted_rating))
        
        return recommendations_with_names
    
    def get_final_recommendations(self, username):
        """
        Get final sentiment-enhanced recommendations for deployment
        
        Args:
            username (str): Username to get recommendations for
        
        Returns:
            dict: Final recommendations with sentiment enhancement
        """
        try:
            # Check if user exists
            if username not in self.user_mappings['user_to_idx']:
                return {
                    'success': False,
                    'error': f"User '{username}' not found in the system",
                    'recommendations': []
                }
            
            # Get top 20 CF recommendations
            cf_recommendations = self.get_user_recommendations(username, 20)
            
            if not cf_recommendations:
                return {
                    'success': False,
                    'error': "No recommendations available for this user",
                    'recommendations': []
                }
            
            # Calculate sentiment scores for each recommended product
            products_with_sentiment = []
            
            for i, (product_name, predicted_rating) in enumerate(cf_recommendations, 1):
                sentiment_info = self.calculate_product_sentiment_score(product_name)
                
                enhanced_rec = {
                    'original_rank': i,
                    'product_name': product_name,
                    'predicted_rating': round(predicted_rating, 2),
                    'total_reviews': sentiment_info['total_reviews'],
                    'positive_percentage': sentiment_info['positive_percentage'],
                    'average_rating': sentiment_info['average_rating'],
                    'sentiment_score': sentiment_info['sentiment_score']
                }
                
                products_with_sentiment.append(enhanced_rec)
            
            # Sort by sentiment score and select top 5
            products_with_sentiment.sort(key=lambda x: x['sentiment_score'], reverse=True)
            final_recommendations = products_with_sentiment[:5]
            
            # Add final ranking
            for i, rec in enumerate(final_recommendations, 1):
                rec['final_rank'] = i
            
            return {
                'success': True,
                'username': username,
                'recommendations': final_recommendations,
                'total_recommendations': len(final_recommendations)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Error generating recommendations: {str(e)}",
                'recommendations': []
            }
    
    def get_available_users(self, limit=100):
        """
        Get list of available users in the system
        
        Args:
            limit (int): Maximum number of users to return
        
        Returns:
            list: List of available usernames
        """
        return list(self.user_mappings['users'])[:limit]
    
    def is_valid_user(self, username):
        """
        Check if a username exists in the system
        
        Args:
            username (str): Username to check
        
        Returns:
            bool: True if user exists, False otherwise
        """
        return username in self.user_mappings['user_to_idx']


# Global instance of the recommendation system
recommendation_system = SentimentBasedRecommendationSystem()

# Initialize the system
def initialize_system():
    """Initialize the recommendation system"""
    success = recommendation_system.load_models()
    if success:
        print("Recommendation system initialized successfully!")
    else:
        print("Failed to initialize recommendation system!")
    return success

# Main functions for Flask app
def get_recommendations_for_user(username):
    """Main function to get recommendations for Flask app"""
    return recommendation_system.get_final_recommendations(username)

def get_valid_users(limit=50):
    """Get valid users for Flask app"""
    return recommendation_system.get_available_users(limit)

def validate_user(username):
    """Validate user for Flask app"""
    return recommendation_system.is_valid_user(username)
