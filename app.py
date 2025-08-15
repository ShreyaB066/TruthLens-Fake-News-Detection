import streamlit as st
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths relative to repo root
model_path = os.path.join("model", "fake_news_model.pkl")
vectorizer_path = os.path.join("model", "vectorizer.pkl")

# Load model + vectorizer
with open(model_path, "rb") as mf:
    model = pickle.load(mf)
with open(vectorizer_path, "rb") as vf:
    vectorizer = pickle.load(vf)

st.title("📰 TruthLens: Fake News Detection")
st.write("Enter a news article below to check whether it is **Fake** or **Real**.")

text = st.text_area("📝 News Article Content")

def predict_one(t: str):
    X = vectorizer.transform([t])
    pred = model.predict(X)[0]   # 0 = FAKE, 1 = REAL  (as per your training)
    return int(pred)

if st.button("Detect"):
    if not text.strip():
        st.warning("⚠️ Please enter a news article.")
    else:
        y = predict_one(text)
        if y == 0:
            st.error("🛑 This news is predicted to be **FAKE**.")
        else:
            st.success("✅ This news is predicted to be **REAL**.")
