import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

news_data = pd.read_csv('C:\\Users\\happy\\OneDrive\\Desktop\\ml project\\prog\\train.csv')

news_dataset = news_data.sample(frac=0.1, random_state=42)
# replacing the null values with empty string
news_dataset = news_dataset.fillna('')

# merging the author name and news title
news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']

X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']



port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

chunk_size = 1000
num_chunks = len(news_dataset) // chunk_size

for i in range(num_chunks + 1):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(news_dataset))

    news_dataset.iloc[start_idx:end_idx, news_dataset.columns.get_loc('content')] = news_dataset.iloc[start_idx:end_idx]['content'].apply(stemming)


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(news_dataset['content'])

# Separating the data & label
Y = news_dataset['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Model training
model = LogisticRegression()
model.fit(X_train, Y_train)

# Accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)

# Accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)

# Take input from the user
user_input_author = input("Enter the author name: ")
user_input_title = input("Enter the news title: ")

# Create a DataFrame with the user input
user_input_data = pd.DataFrame({
    'author': [user_input_author],
    'title': [user_input_title],
})

# Merge author name and title
user_input_data['content'] = user_input_data['author'] + ' ' + user_input_data['title']

# Apply stemming
user_input_data['content'] = user_input_data['content'].apply(stemming)

# Vectorize the user input
user_input_X = vectorizer.transform(user_input_data['content'])

# Make predictions
user_predictions = model.predict(user_input_X)

# Display the predictions
if user_predictions[0] == 0:
    print('Prediction: The news is Real')
else:
    print('Prediction: The news is Fake')
