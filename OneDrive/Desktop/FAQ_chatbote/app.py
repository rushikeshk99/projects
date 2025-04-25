from flask import Flask, request, render_template
import json
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# FAQ डेटा लोड करा
with open('FAQ.json') as f:
    data = json.load(f)

questions = [item['question'] for item in data]
answers = [item['answer'] for item in data]

nltk.download('punkt')
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_input = request.form["msg"]
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    idx = np.argmax(similarity)

    if similarity[0][idx] > 0.3:
        return answers[idx]
    else:
        return "माफ करा, मला याचं उत्तर माहिती नाही."

if __name__ == "__main__":
    app.run(debug=True)
