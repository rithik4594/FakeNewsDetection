# Fake News Detection

A Machine Learning project that detects whether a news article is real or fake using Natural Language Processing (NLP) techniques and supervised learning algorithms.

## Project Overview

Fake news has become a major challenge in the digital age. This project aims to classify news articles as **Real** or **Fake** by analyzing their textual content using machine learning and natural language processing techniques.

## Features

- Text preprocessing and cleaning
- Natural Language Processing (NLP)
- Feature extraction using TF-IDF
- Machine Learning model training
- News classification (Real/Fake)
- Model performance evaluation
- Prediction on custom news articles

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- Jupyter Notebook

## Dataset

This project uses publicly available Fake News datasets containing:

- News Title
- News Text
- Subject/Category
- Publication Date
- Label (Real/Fake)

Example datasets:
- Fake.csv
- True.csv

## Project Structure

```
FakeNewsDetection/
│
├── dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── notebooks/
├── models/
├── FakeNewsDetection.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/FakeNewsDetection.git
```

### 2. Navigate to the Project Directory

```bash
cd FakeNewsDetection
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

Run the Jupyter Notebook:

```bash
jupyter notebook
```

Open `FakeNewsDetection.ipynb` and execute the cells to train the model and test predictions.

## Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Text Preprocessing
4. Feature Extraction (TF-IDF)
5. Model Training
6. Model Evaluation
7. Fake News Prediction

## Model Performance

The model is evaluated using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Sample Prediction

Input:
```
Scientists discover a new renewable energy source that reduces electricity costs by 50%.
```

Output:
```
Real News
```

## Future Improvements

- Deep Learning Models (LSTM, BERT)
- Real-time News Verification
- Web Application using Streamlit
- Browser Extension for Fake News Detection

## Author

**Rithik S**

B.Tech – Artificial Intelligence and Data Science  
Karpagam Institute of Technology

## License

This project is licensed under the MIT License.
