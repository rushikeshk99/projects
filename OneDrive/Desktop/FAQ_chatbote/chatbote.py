import json
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQ data
with open('FAQ.json') as f:
    data = json.load(f)

questions = [item['question'] for item in data]
answers = [item['answer'] for item in data]

# Preprocess
nltk.download('punkt')
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Chat loop
print("Hi! Ask me something (type 'exit' to quit):")
while True:
    user_input = input("You: ")
    
    if user_input.lower() == 'exit':
        print("Bye!")
        break
    elif user_input.lower() == 'hello':
        print("Bot: Tell me how can I help you?")
        continue
    elif user_input.lower() == 'ok':
        print("Bot: Is there anything else I can help you with?")
        continue
    
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    idx = np.argmax(similarity)

    if similarity[0][idx] > 0.05:  # Lowered threshold
        print("Bot:", answers[idx])
    else:
        print("Bot: Sorry, I don’t have an answer for that.")

        continue

    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)
    idx = np.argmax(similarity)

    # Lower the threshold to make it more responsive
    if similarity[0][idx] > 0.1:  # Adjust the threshold here (0.1 is more lenient)
        print("Bot:", answers[idx])
    else:
        print("Bot: Sorry, I don’t have an answer for that.")
