
import streamlit as st
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim.models import Word2Vec
from tensorflow.keras.models import load_model


# Page configuration
st.set_page_config(
    page_title="Spam Message Detection",
    page_icon="📩",
    layout="centered"
)

# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    return True

download_nltk_resources()


# Load trained models
@st.cache_resource
def load_models():
    word2vec_model = Word2Vec.load("word2vec_model.model")
    ann_model = load_model("spam_ann_model.keras")

    return word2vec_model, ann_model


word2vec_model, ann_model = load_models()


# Text preprocessing
def clean(doc):

    regex = '[^a-zA-Z.]'
    doc = re.sub(regex, " ", doc)
    doc = doc.lower()

    tokens = nltk.word_tokenize(doc)

    stop_words = set(stopwords.words('english'))

    filtered_tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    lemmatizer = WordNetLemmatizer()

    lemmatized_tokens = [
        lemmatizer.lemmatize(token)
        for token in filtered_tokens
    ]

    return " ".join(lemmatized_tokens)


# Convert message into Word2Vec vector
def get_sentence_vector(words):

    vectors = [
        word2vec_model.wv[word]
        for word in words
        if word in word2vec_model.wv
    ]

    if len(vectors) == 0:
        return np.zeros(100)

    return np.mean(vectors, axis=0)


# Predict message
def predict_message(message):

    cleaned = clean(message)
    words = cleaned.split()

    vector = get_sentence_vector(words).reshape(1, 100)

    prediction = ann_model.predict(vector, verbose=0)[0][0]

    if prediction > 0.5:
        return "Spam", float(prediction)

    else:
        return "Ham", float(1 - prediction)


# Streamlit UI
st.title("📩 Spam Message Detection")

st.write(
    "Enter a message below to check whether it is Spam or Ham."
)

message = st.text_area(
    "Enter your message:",
    placeholder="Type your message here..."
)

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:
        result, confidence = predict_message(message)

        if result == "Spam":
            st.error("🚨 This message is SPAM!")

        else:
            st.success("✅ This message is HAM!")

        st.write(
            f"**Prediction Confidence:** {confidence * 100:.2f}%"
        )
