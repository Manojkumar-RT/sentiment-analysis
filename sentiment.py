from datasets import load_dataset
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load IMDb dataset
data = load_dataset("imdb", split="train[:2000]")  # use first 2000 rows to keep it fast
df = pd.DataFrame(data)

# Rename columns for clarity
df = df.rename(columns={"text": "review", "label": "sentiment"})

# Preprocess reviews
stop_words = set(stopwords.words('english'))
df['review'] = df['review'].str.lower()
df['review'] = df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Vectorize
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

# Split & Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc * 100:.2f}%")

# Custom Prediction
sample = ["This product is terrible and a complete waste of money."]
sample_vector = vectorizer.transform(sample)
result = model.predict(sample_vector)[0]
print("🧠 Prediction:", "Positive" if result == 1 else "Negative")
