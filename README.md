# Email Spam Classification

This project demonstrates an email spam classification model using the Naive Bayes algorithm. The goal is to classify emails as spam or not spam based on their content.

![image](https://github.com/user-attachments/assets/8146b318-4463-427d-9232-8067fc9189f9)




## Project Summary

1. **Data Preparation**
   - Loaded and cleaned the email dataset.
   - Preprocessed the text data by converting it to lowercase, removing punctuation, replacing non-alphabetic characters with spaces, removing stop words, tokenizing, and stemming.

2. **Feature Extraction**
   - Utilized the Bag of Words model to transform text data into numerical features for machine learning.

3. **Model Training and Evaluation**
   - Trained a Naive Bayes classifier on the processed text data.
   - Evaluated the model using accuracy, precision, recall, and visualized the performance with a confusion matrix.

4. **Streamlit Application**
   - Developed a Streamlit app to allow users to input email text and receive a prediction on whether it is spam or not.
   - The app processes the input text in real-time and uses the trained Naive Bayes model to make predictions.

## Getting Started

### Prerequisites

- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, nltk, scikit-learn, streamlit

### Installation

1. Clone the repository:
    ```sh
   https://github.com/Prasadayus/Email-Spam-Classification.git
    ```

2. Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```

### Running the Streamlit App

1. Start the Streamlit app:
    ```sh
    streamlit run spam prediction.py
    ```

2. Open your web browser and go to the URL provided by Streamlit to interact with the app.

## Model Pickling

- The trained Naive Bayes model and vectorizer are pickled and saved for future use. You can load these files to make predictions without needing to retrain the model.

## Usage

1. Enter an email text in the provided text area.
2. Click on "Submit" to get a prediction on whether the email is spam or not.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
