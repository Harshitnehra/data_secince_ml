from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__, 
            template_folder="../frontend/templates",
            static_folder="../frontend/static")

# Load saved models
model = pickle.load(open("emotion_pipeline.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

def preprocess(text):
    """Exact match to training preprocessing."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)       # Remove punctuation
    text = re.sub(r'\d+', '', text)           # Remove digits
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"].strip()
    if not text:
        return render_template("index.html", prediction_text="Please enter some text!")
    
    processed_text = preprocess(text)
    prediction = model.predict([processed_text])  # Pipeline handles TF-IDF
    result = le.inverse_transform(prediction)[0]  # Decode emotion name
    
    return render_template("index.html", prediction_text=f"Detected Emotion: **{result}** 🎭")

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
