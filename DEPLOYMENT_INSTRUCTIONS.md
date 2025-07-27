
# Sentiment-Based Product Recommendation System - Deployment Instructions

## Prerequisites
- Python 3.8 or higher
- pip package manager

## Installation Steps

1. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify all model files are present:**
   - final_sentiment_model.pkl
   - tfidf_vectorizer.pkl (or count_vectorizer.pkl)
   - label_encoder.pkl
   - final_recommendation_system.pkl
   - user_mappings.pkl
   - item_mappings.pkl
   - rating_matrix.npy
   - preprocessed_data.csv

3. **Run the Flask application:**
   ```bash
   python app.py
   ```

4. **Access the application:**
   - Open your web browser
   - Navigate to: http://localhost:9000
   - Enter a username and get recommendations!

## File Structure
```
project/
├── app.py                              # Flask web application
├── model.py                            # ML models and recommendation logic
├── requirements.txt                    # Python dependencies
├── templates/                          # HTML templates
│   ├── base.html                      # Base template
│   ├── index.html                     # Home page
│   ├── results.html                   # Results page
│   ├── about.html                     # About page
│   └── error.html                     # Error page
├── final_sentiment_model.pkl          # Trained sentiment model
├── tfidf_vectorizer.pkl              # Text vectorizer
├── label_encoder.pkl                  # Label encoder
├── final_recommendation_system.pkl    # Recommendation system
├── user_mappings.pkl                  # User ID mappings
├── item_mappings.pkl                  # Item ID mappings
├── rating_matrix.npy                  # User-item rating matrix
└── preprocessed_data.csv              # Processed dataset
```

## API Endpoints
- GET /                                # Home page
- POST /recommend                      # Get recommendations (form)
- GET /api/recommend/<username>        # Get recommendations (API)
- GET /api/users                       # Get valid users
- GET /api/validate/<username>         # Validate user
- GET /about                           # About page
- GET /health                          # Health check

## Usage
1. Enter a valid username on the home page
2. Click "Get My Recommendations"
3. View your personalized top 5 product recommendations
4. Each recommendation shows:
   - Product name
   - Predicted rating
   - Sentiment score
   - Positive review percentage
   - Total number of reviews

## Troubleshooting
- If you get "User not found" error, try selecting from the dropdown list
- If the system fails to initialize, ensure all .pkl files are present
- For performance issues, the system processes recommendations in real-time

## System Requirements
- RAM: Minimum 4GB (8GB recommended)
- Storage: ~500MB for model files and dependencies
- Network: Internet connection for initial package installation
