# fake_news_detector.py
# Run in VS Code terminal: python fake_news_detector.py

from flask import Flask, render_template_string, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Step 1: Example Dataset
# -----------------------------
data = {
    "headline": [
        "Government launches new healthcare scheme",
        "Aliens landed in New York City",
        "Stock market hits record high",
        "Celebrity claims to time travel",
        "Scientists discover cure for rare disease",
        "Man says he spoke to ghosts"
    ],
    "label": ["Real", "Fake", "Real", "Fake", "Real", "Fake"]
}
df = pd.DataFrame(data)

# -----------------------------
# Step 2: Train Model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['headline'], df['label'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# -----------------------------
# Step 3: Flask Web App
# -----------------------------
app = Flask(__name__)

# Styled HTML template (like Career Roadmap)
index_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detector</title>
    <style>
        body {
            font-family: Arial;
            background: linear-gradient(to right, #667eea, #764ba2);
            margin: 0;
            color: white;
        }
        .container {
            width: 400px;
            margin: 100px auto;
            background: white;
            color: black;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
        }
        input[type="text"], input[type="submit"] {
            width: 100%;
            padding: 10px;
            margin: 10px 0;
            border-radius: 8px;
        }
        input[type="submit"] {
            background: #667eea;
            color: white;
            border: none;
            cursor: pointer;
        }
    </style>
</head>
<body>
<div class="container">
    <h2>📰 Fake News Detector</h2>
    <form action="/predict" method="post">
        <label>Enter News Headline</label>
        <input type="text" name="headline" placeholder="Type headline here">
        <input type="submit" value="Check Headline 🚀">
    </form>
</div>
</body>
</html>
"""

result_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Result</title>
    <style>
        body {
            font-family: Arial;
            background: linear-gradient(to right, #667eea, #764ba2);
            margin: 0;
            color: white;
        }
        .container {
            width: 400px;
            margin: 100px auto;
            background: white;
            color: black;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
        }
        a {
            display: inline-block;
            margin-top: 15px;
            text-decoration: none;
            color: #667eea;
        }
    </style>
</head>
<body>
<div class="container">
    <h2>Prediction Result</h2>
    <p><strong>Headline:</strong> {{ headline }}</p>
    <p><strong>Classification:</strong> {{ prediction }}</p>
    <a href="/">🔙 Try another</a>
</div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(index_html)

@app.route('/predict', methods=['POST'])
def predict():
    headline = request.form['headline']
    features = vectorizer.transform([headline])
    prediction = model.predict(features)[0]
    return render_template_string(result_html, headline=headline, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
