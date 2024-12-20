from flask import Flask, render_template, request, jsonify
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)

# Load the model and tokenizer
model_path = r'C:\Users\jeffr\OneDrive\Desktop\Data_code\sentiment_model'
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_sentiment(input_text):
    # Tokenize the input text
    encoding = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    encoding = {key: val.to(device) for key, val in encoding.items()}
    
    # Run the model
    with torch.no_grad():
        outputs = model(**encoding)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
    
    # Ensure the output is in the expected format (3 classes)
    if len(probabilities) != 3:
        raise ValueError("Unexpected number of output classes. Expected 3 probabilities.")
    
    # Convert probabilities to regular Python floats
    probabilities = [float(prob) for prob in probabilities]
    
    # Get the predicted class index and sentiment label
    predicted_class = torch.argmax(torch.tensor(probabilities), dim=-1).item()
    sentiment = ['Negative', 'Neutral', 'Positive'][predicted_class]
    
    return sentiment, probabilities

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/anal.html')
def analysis_page():
    return render_template('anal.html')

@app.route('/analyze')
def analyze():
    text = request.args.get('text')
    
    try:
        sentiment, probabilities = predict_sentiment(text)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
    # Send sentiment and probabilities as JSON
    return jsonify({
        'result': {
            'text': f"Predicted sentiment: {sentiment}",
            'type': sentiment.lower(),
            'probabilities': {
                'negative': probabilities[0],
                'neutral': probabilities[1],
                'positive': probabilities[2]
            }
        }
    })

if __name__ == "__main__":
    app.run(debug=True)
