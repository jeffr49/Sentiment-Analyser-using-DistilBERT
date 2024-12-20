# Sentiment-Analyser-using-DistilBERT

Dataset: https://drive.google.com/file/d/1AUcoTg2S8nUC3kDLOgNReAGXP1cXy2gR/view?usp=sharing  

Trained model: https://drive.google.com/drive/folders/1YEmm0SniMUmwC--PedwjBorrZZTkhVnJ?usp=sharing

This project is a Sentiment Analysis Web Application designed to analyze the sentiment of input text. It uses a trained sentiment analysis model to classify text into positive, neutral, or negative categories with associated probabilities. This web app provides an easy-to-use interface for users to input text and view the sentiment analysis result in real-time.

Sentiment_analysis.ipynb contains the code in which the model was trained

# Features
Sentiment Analysis: The core feature of the app is analyzing text and classifying it as positive, neutral, or negative with corresponding probabilities.
Real-time Results: Users can see the analysis result immediately after entering the text.
Interactive UI: The application has a clean and user-friendly interface with a responsive layout.
Back Button: A simple navigation button to go back and enter a new text for analysis.
Result Display: Displays the input text in bold, the sentiment result, and the sentiment probabilities clearly formatted for the user.
Why use this?
This sentiment analysis tool can be used for:

Customer Feedback Analysis: Quickly analyze reviews, comments, or feedback to determine customer sentiment.
Social Media Sentiment Tracking: Monitor the sentiment of social media posts or tweets on specific topics or brands.
Text Analytics: Gain insights from large amounts of text data by automatically classifying sentiments.
The tool is simple to use, and with its real-time results and clear presentation, it helps users make data-driven decisions with ease.

# Why we built this?
The need for sentiment analysis tools is growing rapidly as companies and individuals strive to understand public opinion and feedback in an efficient way. With the massive amounts of text data produced daily, it's important to have a tool that can quickly identify the sentiment of this data without manually reading each piece of content.

This project was built to provide an easy-to-use web interface for sentiment analysis, making it accessible to a wide range of users without requiring them to be familiar with machine learning models. By combining NLP (Natural Language Processing) with a sleek web interface, the app helps users gain insights from textual data without having to be experts in data science.

# Deployment instructions
# Prerequisites
Before running the application, make sure you have the following installed:

Python 3.x (preferably the latest version)  
Flask: Python web framework to run the server  
TensorFlow or PyTorch (for the sentiment analysis model)  
scikit-learn (if you're using additional tools for model preprocessing)  
Numpy and Pandas (for data manipulation)  

