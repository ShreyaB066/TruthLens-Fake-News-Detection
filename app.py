import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# File paths (adjust if your .pkl files are in a folder)
model_path = "fake_news_model.pkl"
vectorizer_path = "vectorizer.pkl"

# Load the trained model and vectorizer
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit app title and intro
st.title("📰 TruthLens: Fake News Detection")
st.write("Enter a news article below to check whether it is **Fake** or **Real**.")

# Text input
input_text = st.text_area("📝 News Article Content")

# Prediction function
def predict(text):
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)
    return int(prediction[0])  # 0 = Fake, 1 = Real

# Predict button
if st.button("Detect"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter a news article.")
    else:
        result = predict(input_text)
        if result == 0:  # Fake news
            st.error("🛑 This news is predicted to be **FAKE**.")
        else:  # Real news
            st.success("✅ This news is predicted to be **REAL**.")
