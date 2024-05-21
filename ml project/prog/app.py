from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from time import sleep

app = Flask(__name__)

# Load or preprocess your data as needed
news_data = pd.read_csv('train.csv')
news_dataset = news_data.sample(frac=0.5, random_state=42)
news_dataset = news_dataset.fillna('')
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

# Separating the data & label
X = news_dataset['content']
Y = news_dataset['label']

# Initialize the vectorizer
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split the data for training the model
X_train, X_test, Y_train, Y_test = train_test_split(X_vectorized, Y, test_size=0.2, stratify=Y, random_state=2)

# Initialize the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Function for stemming
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        author = request.form['author']
        title = request.form['title']

        # Create a DataFrame with the user input
        user_input_data = pd.DataFrame({
            'author': [author],
            'title': [title],
        })

        # Merge author name and title
        user_input_data['content'] = user_input_data['author'] + ' ' + user_input_data['title']

        # Apply stemming
        user_input_data['content'] = user_input_data['content'].apply(stemming)

        # Vectorize the user input
        user_input_X = vectorizer.transform(user_input_data['content'])

        # Make predictions
        user_predictions_proba = model.predict_proba(user_input_X)

        # Calculate fake percentage (assuming the second class is 'fake')
        fake_percentage = user_predictions_proba[0, 1] * 100

        # Display the predictions
        if fake_percentage > 50:
            prediction = 'The news is Fake'
        else:
            prediction = 'The news is Real'

        # Add a delay for 5 seconds (adjust as needed)
        sleep(5)

        return render_template('index.html', prediction=prediction, fake_percentage=fake_percentage)

if __name__ == '__main__':
    app.run(debug=True)
