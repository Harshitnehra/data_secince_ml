import pandas as pd
import re
import nltk  # Imported but optional (not used directly)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import pickle

# Load dataset (adjust path if needed)
df = pd.read_csv('train.txt', sep=';', header=None, names=['text', 'emotion'])

# Label encode emotions
le = LabelEncoder()
df['emotion'] = le.fit_transform(df['emotion'])
print("Emotion classes:", le.classes_)  # Debug: see mapping

# Basic text cleaning (exact same as Flask)
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)       # remove punctuation
    text = re.sub(r'\d+', '', text)           # remove digits
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

df['text'] = df['text'].apply(preprocess)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['emotion'], test_size=0.2, random_state=42
)

# Pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Save pipeline and LabelEncoder
pickle.dump(pipeline, open("emotion_pipeline.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("Pipeline and LabelEncoder saved successfully ✅")
