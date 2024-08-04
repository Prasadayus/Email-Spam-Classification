import streamlit as st
import pandas as pd
import pickle
import string
import re
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

st.write("""
# *Email Spam Classification*

**This app predicts whether the Email is Spam or not**
""")

st.subheader("Enter your Email to know whether it is Spam or not")
input_text = st.text_area("Enter your Email here")

if st.button("Submit"):

    # Converting to lowercase
    input_text = input_text.lower()
    
    # Removing punctuation
    exclude = string.punctuation
    input_text = input_text.translate(str.maketrans('', '', exclude))
    
    # Remove non-alphabetic characters
    input_text = re.sub(r'[^a-zA-Z\s]', ' ', input_text)
    
    # Downloading NLTK resources
    nltk.download('stopwords')
    nltk.download('punkt')
    
    # Remove stopwords
    stop_words = stopwords.words('english')
    def remove_stopwords(text):
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)
    
    input_text = remove_stopwords(input_text)
    
    # Tokenize
    input_text = word_tokenize(input_text)
    
    # Stemming
    snowballstemmer = SnowballStemmer('english')
    input_text = [snowballstemmer.stem(token) for token in input_text]
    input_text = ' '.join(input_text)
    
    # Load the model and vectorizer
    with open('naive_bayes_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
        
    # Vectorize the input text
    input_text_vectorized = vectorizer.transform([input_text])  
    
    prediction = model.predict(input_text_vectorized)
    
    if prediction[0] == 1:
        st.write("Your E-mail is ***Spam***.")
    else:
        st.write("The email is classified as ***Not Spam***.")
