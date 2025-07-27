"""
Sentiment-Based Product Recommendation System - Flask Web Application

This Flask application provides a web interface for the sentiment-based
product recommendation system.

Author: Capstone Project
Date: 2025
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import sys
from model import initialize_system, get_recommendations_for_user, get_valid_users, validate_user

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'sentiment_recommendation_system_2025'  # Change this in production

# Global variable to track system initialization
system_initialized = False

def initialize_app():
    """Initialize the recommendation system"""
    global system_initialized
    if not system_initialized:
        print("Initializing recommendation system...")
        system_initialized = initialize_system()
        if system_initialized:
            print("✓ Recommendation system ready!")
        else:
            print("✗ Failed to initialize recommendation system!")
    return system_initialized

# Initialize system at startup
with app.app_context():
    initialize_app()

@app.route('/')
def index():
    """Home page"""
    if not system_initialized:
        return render_template('error.html', 
                             error_message="System not initialized. Please try again later.")
    
    # Get sample users for the dropdown
    sample_users = get_valid_users(50)
    return render_template('index.html', sample_users=sample_users)

@app.route('/recommend', methods=['POST'])
def recommend():
    """Handle recommendation request"""
    if not system_initialized:
        flash('System not initialized. Please try again later.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Get username from form
        username = request.form.get('username', '').strip()
        
        if not username:
            flash('Please enter a username.', 'error')
            return redirect(url_for('index'))
        
        # Validate user
        if not validate_user(username):
            flash(f'User "{username}" not found in the system. Please select a valid user.', 'error')
            return redirect(url_for('index'))
        
        # Get recommendations
        result = get_recommendations_for_user(username)
        
        if not result['success']:
            flash(f'Error: {result["error"]}', 'error')
            return redirect(url_for('index'))
        
        # Render results page
        return render_template('results.html', 
                             username=username,
                             recommendations=result['recommendations'],
                             total_recommendations=result['total_recommendations'])
    
    except Exception as e:
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/recommend/<username>')
def api_recommend(username):
    """API endpoint for recommendations"""
    if not system_initialized:
        return jsonify({
            'success': False,
            'error': 'System not initialized'
        }), 500
    
    try:
        # Validate user
        if not validate_user(username):
            return jsonify({
                'success': False,
                'error': f'User "{username}" not found in the system'
            }), 404
        
        # Get recommendations
        result = get_recommendations_for_user(username)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/api/users')
def api_users():
    """API endpoint to get valid users"""
    if not system_initialized:
        return jsonify({
            'success': False,
            'error': 'System not initialized'
        }), 500
    
    try:
        limit = request.args.get('limit', 50, type=int)
        users = get_valid_users(limit)
        return jsonify({
            'success': True,
            'users': users,
            'total_users': len(users)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/api/validate/<username>')
def api_validate(username):
    """API endpoint to validate a user"""
    if not system_initialized:
        return jsonify({
            'success': False,
            'error': 'System not initialized'
        }), 500
    
    try:
        is_valid = validate_user(username)
        return jsonify({
            'success': True,
            'username': username,
            'is_valid': is_valid
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if system_initialized else 'unhealthy',
        'system_initialized': system_initialized
    })

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('error.html', 
                         error_message="Page not found."), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('error.html', 
                         error_message="Internal server error. Please try again later."), 500

if __name__ == '__main__':
    # Check if all required files exist
    required_files = [
        'final_sentiment_model.pkl',
        'label_encoder.pkl',
        'final_recommendation_system.pkl',
        'user_mappings.pkl',
        'item_mappings.pkl',
        'rating_matrix.npy',
        'preprocessed_data.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("Error: Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease run the Jupyter notebook to generate all required files.")
        sys.exit(1)
    
    # Check for vectorizer file
    vectorizer_exists = os.path.exists('tfidf_vectorizer.pkl') or os.path.exists('count_vectorizer.pkl')
    if not vectorizer_exists:
        print("Error: Missing vectorizer file (tfidf_vectorizer.pkl or count_vectorizer.pkl)")
        sys.exit(1)
    
    print("All required files found. Starting Flask application...")
    
    # Run the Flask app
    # Get port from environment variable (for cloud deployment) or default to 9000 for local
    port = int(os.environ.get('PORT', 9000))

    # Set debug=False for production deployment (Render doesn't set FLASK_ENV)
    # Debug is only enabled if explicitly set to 'development'
    debug_mode = os.environ.get('FLASK_ENV') == 'development'

    app.run(debug=debug_mode, host='0.0.0.0', port=port)
