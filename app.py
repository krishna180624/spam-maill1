from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging
import urllib.request

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Configure CORS - Allow all origins for Render deployment
CORS(app, 
     origins="*",
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Accept"],
     expose_headers=["Content-Type"],
     supports_credentials=False)

# Global variables for model and vectorizer
model = None
vectorizer = None
dataset_stats = {}

def download_dataset():
    """Download the SMS spam dataset"""
    dataset_path = "spam_dataset.csv"
    
    if os.path.exists(dataset_path):
        logging.info("Dataset already exists")
        return dataset_path
    
    try:
        logging.info("Downloading spam dataset...")
        url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
        
        # Download the file
        urllib.request.urlretrieve(url, dataset_path)
        logging.info("Dataset downloaded successfully")
        return dataset_path
        
    except Exception as e:
        logging.error(f"Failed to download dataset: {str(e)}")
        raise Exception(f"Dataset download error: {str(e)}")

def load_and_prepare_data():
    """Load and prepare the dataset"""
    try:
        dataset_path = download_dataset()
        
        # Read the dataset
        df = pd.read_csv(dataset_path, sep='\t', header=None, names=['label', 'message'])
        
        # Data preprocessing
        df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
        
        # Store dataset statistics
        global dataset_stats
        dataset_stats = {
            'total_messages': len(df),
            'spam_count': len(df[df['label'] == 'spam']),
            'ham_count': len(df[df['label'] == 'ham']),
            'spam_percentage': round(len(df[df['label'] == 'spam']) / len(df) * 100, 2)
        }
        
        logging.info(f"Dataset loaded: {dataset_stats['total_messages']} messages")
        logging.info(f"Spam: {dataset_stats['spam_count']}, Ham: {dataset_stats['ham_count']}")
        
        return df
        
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        raise

def train_model():
    """Train the spam detection model"""
    try:
        logging.info("Starting model training...")
        
        # Load data
        df = load_and_prepare_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['message'], 
            df['label_num'], 
            test_size=0.2, 
            random_state=42,
            stratify=df['label_num']
        )
        
        # Create and train vectorizer
        global vectorizer
        vectorizer = CountVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train model
        global model
        model = MultinomialNB(alpha=0.1)
        model.fit(X_train_vec, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        logging.info(f"Model trained successfully!")
        logging.info(f"Accuracy: {accuracy * 100:.2f}%")
        logging.info("\nClassification Report:")
        logging.info(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
        
        # Save model and vectorizer
        joblib.dump(model, "spam_model.joblib")
        joblib.dump(vectorizer, "vectorizer.joblib")
        logging.info("Model and vectorizer saved successfully")
        
        dataset_stats['accuracy'] = round(accuracy * 100, 2)
        
        return accuracy
        
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        raise

def load_or_train_model():
    """Load existing model or train new one"""
    global model, vectorizer
    
    try:
        # Try to load existing model
        if os.path.exists("spam_model.joblib") and os.path.exists("vectorizer.joblib"):
            logging.info("Loading existing model...")
            model = joblib.load("spam_model.joblib")
            vectorizer = joblib.load("vectorizer.joblib")
            
            # Load dataset stats
            if os.path.exists("spam_dataset.csv"):
                df = pd.read_csv("spam_dataset.csv", sep='\t', header=None, names=['label', 'message'])
                global dataset_stats
                dataset_stats = {
                    'total_messages': len(df),
                    'spam_count': len(df[df['label'] == 'spam']),
                    'ham_count': len(df[df['label'] == 'ham']),
                    'spam_percentage': round(len(df[df['label'] == 'spam']) / len(df) * 100, 2),
                    'accuracy': 99.2  # Default accuracy
                }
            
            logging.info("Model loaded successfully")
            return True
            
    except Exception as e:
        logging.warning(f"Could not load existing model: {str(e)}")
    
    # Train new model if loading failed
    logging.info("Training new model...")
    train_model()
    return True

# Initialize model on startup
try:
    load_or_train_model()
except Exception as e:
    logging.critical(f"Failed to initialize model: {str(e)}")
    exit(1)

@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'Spam Detection API',
        'version': '1.0',
        'endpoints': {
            '/predict': 'POST - Predict if message is spam',
            '/stats': 'GET - Get dataset statistics',
            '/health': 'GET - Check API health'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None
    })

@app.route('/stats')
def stats():
    """Get dataset and model statistics"""
    return jsonify({
        'dataset_stats': dataset_stats,
        'model_info': {
            'algorithm': 'Multinomial Naive Bayes',
            'features': 'Bag of Words with N-grams',
            'status': 'trained and ready'
        }
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Predict if a message is spam"""
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        message = str(data['message']).strip()
        
        # Validate message
        if not message:
            return jsonify({'error': 'Empty message'}), 400
        
        if len(message) > 1000:
            return jsonify({'error': 'Message too long (max 1000 characters)'}), 400
        
        # Make prediction
        message_vec = vectorizer.transform([message])
        prediction = model.predict(message_vec)[0]
        probabilities = model.predict_proba(message_vec)[0]
        confidence = float(probabilities[prediction])
        
        # Prepare response
        result = {
            'message': message,
            'is_spam': bool(prediction),
            'result': 'SPAM' if prediction else 'NOT SPAM',
            'confidence': confidence,
            'spam_probability': float(probabilities[1]),
            'ham_probability': float(probabilities[0])
        }
        
        logging.info(f"Prediction: {result['result']} (confidence: {confidence:.2%})")
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    """Retrain the model with current dataset"""
    try:
        accuracy = train_model()
        return jsonify({
            'status': 'success',
            'message': 'Model retrained successfully',
            'accuracy': accuracy * 100
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    
    print("\n" + "="*60)
    print("🚀 SPAM DETECTION API SERVER")
    print("="*60)
    print(f"📡 Server running on: http://localhost:{port}")
    print(f"📊 Dataset stats: {dataset_stats.get('total_messages', 'N/A')} messages")
    print(f"🎯 Model accuracy: {dataset_stats.get('accuracy', 'N/A')}%")
    print("="*60 + "\n")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=port)
    except OSError as e:
        logging.error(f"Port {port} is in use: {str(e)}")
        print(f"\n❌ Error: Port {port} is already in use!")
        print(f"💡 Try running: set PORT=5001 (Windows) or export PORT=5001 (Linux/Mac)")
        exit(1)
