from flask import Flask, render_template, request
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

print("Loading Dataset...")

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = "FAKE"
true["label"] = "REAL"

# Combine datasets
df = pd.concat([fake, true], ignore_index=True)

# Features and labels
X_text = df["text"]
y = df["label"]

# Convert text to vectors
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X = vectorizer.fit_transform(X_text)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Model...")

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Accuracy
predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    "Model Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


@app.route("/")
def home():

    return render_template(
        "index.html",
        accuracy=round(accuracy * 100, 2)
    )


@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    # Convert user input
    vector = vectorizer.transform([news])

    # Model prediction
    prediction = model.predict(vector)[0]

    # Confidence
    confidence = model.predict_proba(vector).max() * 100

    # Similarity search
    similarities = cosine_similarity(
        vector,
        X
    )

    # Best matching article
    best_match_index = similarities.argmax()

    similarity_score = (
        similarities[0][best_match_index]
    )

    matched_article = df.iloc[
        best_match_index
    ]["text"]

    matched_label = df.iloc[
        best_match_index
    ]["label"]

    # Limit article size
    if len(matched_article) > 500:
        matched_article = (
            matched_article[:500] + "..."
        )

    # Friendly message
    if (
        similarity_score > 0.60
        and matched_label == "REAL"
    ):

        prediction_message = (
            "✅ This article may be correct because it closely matches a trusted article in the dataset."
        )

    elif prediction == "FAKE":

        prediction_message = (
            "⚠ This article appears suspicious and may contain misinformation."
        )

    else:

        prediction_message = (
            "✅ This article appears genuine based on the trained model."
        )

    # Top 5 related articles
    top_indices = similarities.argsort()[0][-6:-1]

    related_news = []

    for i in reversed(top_indices):

        article = df.iloc[i]["text"]

        if len(article) > 250:
            article = article[:250] + "..."

        related_news.append(article)

    return render_template(
        "index.html",
        prediction=prediction,
        prediction_message=prediction_message,
        confidence=round(confidence, 2),
        accuracy=round(
            accuracy * 100,
            2
        ),
        related_news=related_news,
        matched_article=matched_article,
        similarity_score=round(
            similarity_score * 100,
            2
        )
    )


if __name__ == "__main__":
    app.run(debug=True)