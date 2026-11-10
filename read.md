# Spam Message Detection using Word2Vec and ANN

## Project Overview
This project is a Machine Learning-based Spam Message Detection system that classifies text messages into two categories: **Spam** and **Ham (Not Spam)**.

It uses Word2Vec to convert text messages into numerical word embeddings and an Artificial Neural Network (ANN) to predict whether a message is spam or legitimate.

Optuna is used for hyperparameter optimization to identify the best ANN architecture and training parameters. The trained model is deployed using Streamlit to provide an interactive user interface.

## Objectives
- Detect spam and legitimate text messages.
- Convert text data into numerical vectors using Word2Vec.
- Build an Artificial Neural Network for classification.
- Optimize model hyperparameters using Optuna.
- Deploy the trained model using Streamlit.

## Technologies Used
- Python
- NumPy
- Pandas
- NLTK
- Gensim (Word2Vec)
- TensorFlow / Keras
- Scikit-learn
- Optuna
- Streamlit

## Project Workflow
1. Data Collection
2. Data Cleaning and Preprocessing
3. Tokenization and Stopword Removal
4. Lemmatization
5. Train-Test Split
6. Word2Vec Embedding Generation
7. ANN Model Building
8. Hyperparameter Optimization using Optuna
9. Model Training and Evaluation
10. Streamlit Deployment

## Model Performance
- Model: Artificial Neural Network (ANN)
- Feature Extraction: Word2Vec
- Hyperparameter Tuning: Optuna
- Test Accuracy: **96.71%**

## Streamlit Application
The application allows users to:
- Enter a text message.
- Predict whether the message is Spam or Ham.
- View the prediction confidence.

## Project Structure
```text
Spam_Message_Detection/
│
├── app.py
├── requirements.txt
├── spam_ann_model.keras
├── word2vec_model.model
└── README.md
```

## How to Run the Project

### 1. Clone the Repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

