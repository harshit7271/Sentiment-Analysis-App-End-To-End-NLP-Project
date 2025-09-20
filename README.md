# Sentiment-Analysis-App-End-To-End-NLP-Project 😊

## Overview
This project showcases a powerful **Sentiment Analysis** pipeline built with Python using machine learning techniques. It predicts the emotional tone (sentiment) of an input text — whether it's happy, sad, angry, or any other emotion expressed in the text.

---
## 🚀 Live Demo

<a href="https://ggkplbcecaqrhzfwsmr7aw.streamlit.app/" target="_blank">
  <img src="https://img.shields.io/badge/Live%20Demo-Open%20App-brightgreen?style=for-the-badge&logo=streamlit" alt="Live Demo"/>
</a>

## 🔍 What Does It Do?

- Processes text input with state-of-the-art **text preprocessing** including:
  - Lowercasing, punctuation and digit removal
  - Stopword removal and emoji filtering
- Converts text into numerical features using two vectorizers:
  - **TF-IDF Vectorizer**
  - **Count Vectorizer**
- Trains and compares two popular sentiment classifiers:
  - **Logistic Regression**
  - **Multinomial Naive Bayes**
- Provides an **interactive Streamlit web app** to enter text and get real-time sentiment predictions.

---

## 🚀 Features

| Feature                    | Details                                  |
|----------------------------|------------------------------------------|
| Data Size                  | ~16,000 labeled text samples             |
| Sentiment Classes          | Multi-class emotions (happy, sad, etc.) |
| Preprocessing              | Clean, normalize, and vectorize text     |
| Models                    | LogisticRegression, MultinomialNB         |
| Deployment                | Streamlit web application with UI        |

---

## 🛠 Installation & Setup

1. Clone this repository: git clone[ https://github.com/harshit7271/sentiment-analysis-project.git
cd sentiment-analysis-project](https://github.com/harshit7271/Sentiment-Analysis-App-End-To-End-NLP-Project)

2. Install required Python packages:   pip install -r train.txt
3. The required pickle files are:
- `vectorizer.pkl` (TF-IDF vectorizer to transform input text)
- `count_vectorizer.pkl` (Count vectorizer for alternative model)
- `SentimentAnalysis.pkl` (LogisticRegression model)
- `naive_bayes_model.pkl` (Multinomial Naive Bayes model)

4. Run the Streamlit app:  streamlit run SA.py

---

## 📋 Usage

- Enter your text input in the provided textbox.
- Choose the model (`Logistic Regression` or `Naive Bayes`) and vectorizer (`TF-IDF` or `CountVectorizer`).
- View real-time sentiment prediction with confidence score and colored feedback.

---

## 🎯 How It Works

1. **Text Input** is cleaned and preprocessed.
2. The **selected vectorizer** transforms the text into numerical features.
3. The **selected ML classifier** predicts the sentiment label.
4. **Results** are displayed with confidence and sentiment category.

---

## 📁 Project Structure
│
├── train.py                # Script to preprocess data, train models and save pickles


├── predict.py              # Script to load model/vectorizer and perform predictions (CLI or reusable)


├── SA.py                  # Streamlit app for user input and live predictions


├── train.txt               # Raw dataset


├── logistic_model.pkl      # Saved Logistic Regression model


├── naive_bayes_model.pkl   # Saved Naive Bayes model


├── tfidf_vectorizer.pkl    # Saved TF-IDF vectorizer


├── count_vectorizer.pkl    # Saved Count Vectorizer


├── requirements.txt        # Dependencies


├── utils.py                # Reusable utility functions: text cleaning, preprocessing, etc.


└── README.md               # Documentation



---

## 📈 Model Performance

| Model                | Vectorizer     | Accuracy  |
|----------------------|----------------|-----------|
| Logistic Regression   | TF-IDF         | 0.86       |
| Multinomial Naive Bayes | Count Vectorizer | 0.76    |

*(Example accuracies based on training results)*

---

## 🙌 Contributions & Acknowledgements

- Thanks to scikit-learn for robust ML tools.
- Streamlit for fast prototyping of web UI.
- NLTK for text preprocessing utilities.
- Dataset contributors for labeled emotion data.

---

## 📞 Contact

Created by [HARSHIT SINGH] — feel free to open issues or pull requests for suggestions or improvements!

---

## 🎉 Have fun analyzing sentiments with this cool app! 😊






