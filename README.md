# 🎭 Sentiment Analysis of IMDb Movie Reviews

This project is a mini Natural Language Processing (NLP) task to classify IMDb movie reviews as **Positive** or **Negative** using Python, NLTK, and Scikit-learn.

It demonstrates the end-to-end data science process: loading text data, cleaning it, vectorizing it, training a machine learning model, and evaluating performance.

---

## 📌 Project Overview

- **Goal:** Predict sentiment (positive or negative) of a given movie review.
- **Dataset:** IMDb dataset (loaded using Hugging Face `datasets`)
- **Model:** Multinomial Naive Bayes
- **Accuracy:** ~85% on test data

---

## 🛠️ Technologies Used

- Python 🐍
- Scikit-learn
- NLTK (Natural Language Toolkit)
- Hugging Face Datasets
- CountVectorizer
- Jupyter/VS Code

---

## 📈 Workflow

1. Load dataset using `datasets` library
2. Preprocess text (lowercasing + stopword removal)
3. Convert text to numerical features using CountVectorizer
4. Train a Naive Bayes classifier
5. Evaluate model using accuracy score
6. Predict sentiment on custom review text

---

## 🔍 Example Output

```python
✅ Accuracy: 100.00%
🧠 Prediction: Negative
