from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# Initialize the sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text):
    return sentiment_pipeline(text)[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message')
    sentiment = analyze_sentiment(user_input)
    response = {
        'message': user_input,
        'sentiment': sentiment['label'],
        'confidence': sentiment['score']
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
