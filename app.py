import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Customer Sentiment Analyzer", layout="centered")

st.title("Customer Sentiment Analyzer")
st.write(
    "Paste a tweet below and the model will predict whether the sentiment is "
    "**Positive** or **Negative**."
)

model = joblib.load("final_sentiment_model.joblib")

tweet = st.text_area(
    "Tweet text",
    height=120,
    placeholder="Type or paste a tweet here..."
)

if st.button("Analyze"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        proba_pos = model.predict_proba([tweet])[0][1]
        proba_neg = 1 - proba_pos
        pred = model.predict([tweet])[0]

        label = "Positive" if pred == 1 else "Negative"

        st.subheader("Prediction")
        st.write(f"**Sentiment:** {label}")

        st.subheader("Confidence")
        st.dataframe(pd.DataFrame([{
            "Probability Positive": round(proba_pos, 3),
            "Probability Negative": round(proba_neg, 3)
        }]))
