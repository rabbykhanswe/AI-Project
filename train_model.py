import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
import os

print("--- Step 1: Loading Data ---")


if not os.path.exists('spam.csv'):
    print("Error: spam.csv file is missing!")
    exit()


try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
except:
    # If latin-1 fails, try standard utf-8
    df = pd.read_csv('spam.csv', encoding='utf-8')


if 'v1' in df.columns and 'v2' in df.columns:
    df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)


df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})

print("Data Loaded Successfully!")


vectorizer = TfidfVectorizer(stop_words='english')


X = vectorizer.fit_transform(df['message'])


y = df['label_num']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Training the Model... Please wait.")
model = MultinomialNB()
model.fit(X_train, y_train)
print("Model Trained Successfully!")


predictions = model.predict(X_test)

print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")


with open('spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Success! 'spam_model.pkl' and 'vectorizer.pkl' are saved.")