import pandas as pd
import numpy as np
import re
import nltk
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

# Download stopwords if not already present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ======================
# 1. LOAD YOUR DATA
# ======================
def load_data(filepath):
    """Load Twitter sentiment data from CSV"""
    df = pd.read_csv(filepath)
    
    # Check if required columns exist
    if 'clean_text' not in df.columns or 'category' not in df.columns:
        raise ValueError("CSV must contain 'clean_text' and 'category' columns")
    
    # Handle missing values
    df = df.dropna(subset=['clean_text', 'category'])
    
    # Convert category to int (just in case)
    df['category'] = df['category'].astype(int)
    
    print(f"✅ Loaded {len(df)} tweets")
    print(f"📊 Class Distribution:\n{df['category'].value_counts().sort_index()}")
    print(f"   (-1=Negative, 0=Neutral, 1=Positive)\n")
    
    return df['clean_text'].values, df['category'].values

# ======================
# 2. TEXT PREPROCESSING
# ======================
def preprocess_text(text):
    """
    Custom preprocessing for Twitter data
    (You can modify this based on your needs)
    """
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions (@username) and hashtags (#) - keep the text
        text = re.sub(r'@\w+|#', '', text)
        
        # Remove numbers and punctuation (optional - keep if you want)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    return ""

# ======================
# 3. TRAIN THE MODEL
# ======================
def train_model(X_texts, y_labels):
    """
    Train a sentiment analysis model using TF-IDF + Logistic Regression
    """
    print("🔄 Preprocessing texts...")
    X_processed = [preprocess_text(text) for text in X_texts]
    
    print("🔄 Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    
    print(f"   Training set: {len(X_train)} tweets")
    print(f"   Test set: {len(X_test)} tweets\n")
    
    # ==========================================
    # Build Pipeline: TF-IDF + Logistic Regression
    # ==========================================
    print("🔄 Building TF-IDF + Logistic Regression pipeline...")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,          # Limit vocabulary size for performance
            ngram_range=(1, 2),          # Unigrams + Bigrams
            stop_words=list(stop_words), # Remove common English stopwords
            min_df=2,                    # Ignore terms that appear in < 2 docs
            max_df=0.85                  # Ignore terms that appear in > 85% docs (too common)
        )),
        ('clf', LogisticRegression(
            multi_class='multinomial',   # For 3 classes (-1, 0, 1)
            solver='saga',               # Good for multiclass
            max_iter=1000,
            class_weight='balanced',     # Handle any class imbalance
            random_state=42,
            C=1.0                        # Regularization strength
        ))
    ])
    
    print("🔄 Training the model...")
    pipeline.fit(X_train, y_train)
    
    # ==========================================
    # Evaluate on Test Set
    # ==========================================
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Model Training Complete!")
    print(f"📊 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n📋 Classification Report:")
    print(classification_report(
        y_test, y_pred, 
        target_names=['Negative (-1)', 'Neutral (0)', 'Positive (1)']
    ))
    
    print("📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # ==========================================
    # Save the Model and Vectorizer
    # ==========================================
    model_dir = Path('model')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'sentiment_model.pkl'

    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"\n💾 Model saved as '{model_path}'")
    
    return pipeline, accuracy

# ======================
# 4. TEST WITH SAMPLE INPUTS
# ======================
def test_with_sample_tweets(model):
    """Test the model with sample tweets to verify it works"""
    sample_tweets = [
        "I absolutely love Modi's policies! Best PM ever! 🇮🇳",  # Positive
        "This is a complete disaster. Worst government ever.",   # Negative
        "Let's see what happens tomorrow in the election.",      # Neutral
        "Modi is doing great work for the country!",            # Positive
        "Why does it take so many years to get justice?",       # Negative (based on your sample)
    ]
    
    print("\n🧪 Testing with sample tweets:")
    print("-" * 60)
    
    for tweet in sample_tweets:
        cleaned = preprocess_text(tweet)
        sentiment = model.predict([cleaned])[0]
        confidence = model.predict_proba([cleaned]).max()
        
        label_map = {-1: "Negative 😞", 0: "Neutral 😐", 1: "Positive 😊"}
        print(f"Tweet: {tweet}")
        print(f"  → Sentiment: {label_map[sentiment]} (Confidence: {confidence:.2%})")
        print()

# ======================
# 5. MAIN EXECUTION
# ======================
if __name__ == "__main__":
    # Update this path to your actual CSV file
    DATA_PATH = "data/Twitter_Data.csv"
    
    try:
        # Load data
        X, y = load_data(DATA_PATH)
        
        # Train model
        model, acc = train_model(X, y)
        
        # Quick test
        test_with_sample_tweets(model)
        
        print("🎉 All done! Your model is ready for the web app.")
        
    except FileNotFoundError:
        print(f"❌ Error: '{DATA_PATH}' not found. Please check the file path.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")