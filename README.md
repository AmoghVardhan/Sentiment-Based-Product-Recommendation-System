# 🛍️ Sentiment-Based Product Recommendation System

A sophisticated machine learning-powered recommendation system that combines collaborative filtering with sentiment analysis to provide personalized product recommendations. This system analyzes user reviews and ratings to deliver more accurate and contextually relevant product suggestions.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v2.3+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.3+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

- **🤖 Hybrid Recommendation Engine**: Combines collaborative filtering with sentiment analysis for enhanced accuracy
- **📊 Advanced Sentiment Analysis**: Uses machine learning models to analyze product review sentiments
- **🎯 Personalized Recommendations**: Tailored suggestions based on user behavior and preferences
- **🌐 Web Interface**: Clean, responsive Flask web application with Bootstrap UI
- **📱 RESTful API**: JSON API endpoints for integration with other applications
- **⚡ Real-time Processing**: Fast recommendation generation with optimized algorithms
- **📈 Optimized ML Pipeline**: Evaluated multiple models (Logistic Regression, Random Forest, XGBoost, Neural Network, Naive Bayes) and vectorization methods (BOW, TF-IDF, Word2Vec) to select the best performing combination

## 🏗️ System Architecture

The system consists of two main components:

1. **Collaborative Filtering System**: Generates initial recommendations based on user-item interactions
2. **Sentiment Enhancement Layer**: Analyzes product reviews to refine recommendations based on sentiment scores

### Machine Learning Pipeline

```
User Input → Collaborative Filtering → Top 20 Products → Sentiment Analysis → Top 5 Enhanced Recommendations
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM (for model loading)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sentiment-recommendation-system.git
   cd sentiment-recommendation-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```
   *Note: NLTK data will be downloaded automatically on first run*

4. **Access the web interface**
   - **Local Development**: Open your browser and navigate to `http://localhost:9000`
   - **Production/Cloud**: The app will use the port assigned by your hosting platform
   - Enter a username to get personalized recommendations

## 📁 Project Structure

```
sentiment-recommendation-system/
├── app.py                              # Flask web application
├── model.py                            # ML models and recommendation logic
├── requirements.txt                    # Python dependencies
├── DEPLOYMENT_INSTRUCTIONS.md          # Detailed deployment guide
├── Final_sentiment_based_product_recommendation_system.ipynb  # Development notebook
├── templates/                          # HTML templates
│   ├── base.html                      # Base template with Bootstrap
│   ├── index.html                     # Home page
│   ├── results.html                   # Recommendations display
│   ├── about.html                     # About page
│   └── error.html                     # Error handling
├── models/                            # Trained ML models
│   ├── final_sentiment_model.pkl      # Sentiment classification model
│   ├── tfidf_vectorizer.pkl          # Text vectorizer
│   ├── label_encoder.pkl             # Label encoder for sentiments
│   ├── final_recommendation_system.pkl # Collaborative filtering model
│   ├── user_mappings.pkl             # User ID mappings
│   ├── item_mappings.pkl             # Product ID mappings
│   └── rating_matrix.npy             # User-item rating matrix
└── data/                             # Dataset files
    ├── preprocessed_data.csv         # Cleaned dataset
    └── cleaned_data.csv              # Original cleaned data
```

## 🔧 API Endpoints

### Web Interface
- `GET /` - Home page with user input form
- `POST /recommend` - Get recommendations via form submission
- `GET /about` - About page with system information

### REST API
- `GET /api/recommend/<username>` - Get recommendations for a user (JSON)
- `GET /api/users` - Get list of valid users
- `GET /api/validate/<username>` - Validate if user exists
- `GET /health` - Health check endpoint

### Example API Usage

```bash
# For local development (replace with your deployed URL for production)
# Get recommendations for a user
curl http://localhost:9000/api/recommend/john_doe

# Get list of valid users
curl http://localhost:9000/api/users

# Validate a user
curl http://localhost:9000/api/validate/john_doe

# For production deployment on Render:
# curl https://your-app-name.onrender.com/api/recommend/john_doe
```

## 🧠 Machine Learning Models

### 1. Sentiment Analysis Model
- **Algorithm**: Logistic Regression (selected after comprehensive evaluation)
- **Features**: Bag of Words (BOW) vectorized text features
- **Performance**: 96.49% accuracy, 96.65% F1-score on test set
- **Input**: Product review text
- **Output**: Sentiment (Positive/Negative/Neutral) with confidence score
- **Cross-validation**: 96.33% ± 0.32% accuracy across 5 folds

### 2. Collaborative Filtering System
- **Algorithm**: Item-based Collaborative Filtering
- **Similarity Metric**: Cosine similarity
- **Matrix**: User-item rating matrix (24,751 users × 267 items)
- **Performance**: RMSE of 0.9136 (outperformed User-based CF with RMSE 1.6355)
- **Output**: Top 20 product recommendations with predicted ratings

### 3. Sentiment Enhancement
- **Process**: Analyzes reviews for recommended products
- **Scoring**: Combines sentiment percentage (70%) + normalized rating (30%)
- **Output**: Top 5 sentiment-enhanced recommendations

## 📊 Performance Metrics

### Sentiment Analysis Model Performance
- **Best Model**: Logistic Regression with BOW features
- **Accuracy**: 96.49%
- **Precision**: 96.91%
- **Recall**: 96.49%
- **F1-Score**: 96.65%
- **ROC AUC**: 96.17%
- **Specificity**: 81.74%
- **Training Time**: 0.10 seconds
- **Prediction Time**: <0.001 seconds

### Recommendation System Performance
- **Algorithm**: Item-based Collaborative Filtering
- **RMSE**: 0.9136 (Root Mean Squared Error)
- **MAE**: 0.6321 (Mean Absolute Error)
- **Matrix Sparsity**: 99.59%
- **Response Time**: <2 seconds for recommendations
- **System Throughput**: 100+ concurrent users supported

## 🔬 Model Evaluation & Selection

### Sentiment Analysis Model Comparison
The system evaluated 5 different machine learning models with 3 vectorization methods:

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC | Training Time |
|-------|----------|-----------|--------|----------|---------|---------------|
| **Logistic Regression** | **96.49%** | **96.91%** | **96.49%** | **96.65%** | **96.17%** | **0.10s** |
| Random Forest | 97.12% | 97.13% | 97.12% | 96.72% | 96.07% | 0.49s |
| Neural Network | 97.12% | 97.00% | 97.12% | 96.81% | 94.81% | 173.98s |
| XGBoost | 93.29% | 95.99% | 93.29% | 94.21% | 96.45% | 0.62s |
| Naive Bayes | 94.50% | 95.95% | 94.50% | 95.03% | 95.31% | 0.00s |

**Selection Rationale**: Logistic Regression was selected using a multi-criteria approach considering:
- Composite score (weighted average of all metrics)
- Sensitivity-specificity balance
- Multi-criteria decision analysis (MCDA)
- Traditional F1-score evaluation

### Vectorization Method Comparison
| Method | Accuracy | Features | Sparsity |
|--------|----------|----------|----------|
| **Bag of Words (BOW)** | **97.56%** | 5,000 | 99.57% |
| TF-IDF | 97.24% | 5,000 | 99.57% |
| Word2Vec | 96.97% | 100 (dense) | N/A |

### Recommendation System Comparison
| System | RMSE | MAE | MSE | Predictions |
|--------|------|-----|-----|-------------|
| User-Based CF | 1.6355 | 1.4455 | 2.6748 | 1,000 |
| **Item-Based CF** | **0.9136** | **0.6321** | **0.8346** | **1,000** |

## 🛠️ Development

### Running in Development Mode

```bash
# Enable debug mode
export FLASK_ENV=development
export FLASK_DEBUG=1
python app.py
```

### Model Training & Evaluation

The system includes pre-trained models, but you can explore the complete model development process using the Jupyter notebook:

```bash
jupyter notebook Final_sentiment_based_product_recommendation_system.ipynb
```

**The notebook includes:**
- Data preprocessing and cleaning
- Feature extraction (BOW, TF-IDF, Word2Vec)
- Comprehensive model evaluation (5 ML algorithms)
- Hyperparameter tuning with GridSearchCV
- Cross-validation and performance analysis
- Recommendation system development and comparison
- Model selection using multiple criteria

## 🚀 Deployment

### Deploying to Render

The application is configured for easy deployment to Render:

1. **Connect your GitHub repository** to Render
2. **Create a new Web Service** with these settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Environment**: Python 3
3. **Environment Variables** (optional):
   - `FLASK_ENV=development` (only if you want debug mode enabled)
   - **Note**: Render automatically sets `PORT`, but not `FLASK_ENV`
4. **Port Configuration**: The app automatically detects Render's assigned port

**Note**: The app will automatically:
- Use Render's assigned port (Render sets `PORT` environment variable)
- Run in production mode (debug disabled by default)
- Download NLTK data on first startup

### Other Cloud Platforms

For deployment to other platforms (Heroku, AWS, etc.), ensure:
- The platform sets the `PORT` environment variable
- Python 3.8+ is available
- All dependencies in `requirements.txt` are installed

⭐ **Star this repository if you found it helpful!**
