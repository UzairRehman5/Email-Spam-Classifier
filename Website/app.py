import nltk
import pickle
import streamlit as st
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from PIL import Image


nltk.download('punkt')

tfidf = pickle.load(open('Website/Models/vectorizer.pkl', 'rb'))
model = pickle.load(open('Website/Models/model.pkl', 'rb'))

logo = Image.open('Website/Images/logo_img.png')
st.set_page_config(page_title='SpamGuard', page_icon=logo)

st.title('SpamGuard')
st.header('Advanced Email/SMS Spam Classifier')

img = Image.open('Website/Images/spam.jpg')
st.image(img, width=250)

input_msg = st.text_input('Enter the message')


# 1. Preprocess

def transform_text(text):
    # Lower casing
    text = text.lower()

    # Tokenization
    text = nltk.word_tokenize(text)

    # Removing HTML tags
    text = [BeautifulSoup(word, 'html.parser').get_text() for word in text]

    # Removing special characters
    text = [i for i in text if i.isalnum()]

    # Removing stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    text = [i for i in text if i not in stop_words and i not in string.punctuation]

    # Stemming
    ps = PorterStemmer()
    text = [ps.stem(i) for i in text]

    return " ".join(text)


if st.button('Predict'):

    transform_msg = transform_text(input_msg)

    # 2. Vectorize

    vectorized_msg = tfidf.transform([transform_msg])

    # 3. Predict

    result = model.predict(vectorized_msg)[0]

    # 4. Display

    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')
