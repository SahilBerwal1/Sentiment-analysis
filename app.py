import streamlit as st
import joblib

model = joblib.load("best_sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("Sentiment Analysis")

text = st.text_area("Enter text")

if st.button("Predict"):
    transformed = vectorizer.transform([text])
    result = model.predict(transformed)
    st.write(result[0])