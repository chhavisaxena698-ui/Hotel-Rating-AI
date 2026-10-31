
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import numpy as np

# Load dataset
data = pd.read_csv("tripadvisor_hotel_reviews.csv")

# Clean data
data = data.dropna(subset=["Review", "Rating"])
data["Review"] = data["Review"].astype(str)
data["Rating"] = pd.to_numeric(data["Rating"], errors="coerce")
data = data.dropna(subset=["Rating"])

# Input and output
X = data["Review"]
y = data["Rating"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
    ("regression", LinearRegression())
])

# Train model
model.fit(X_train, y_train)

# Predict ratings
predictions = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# Save model
joblib.dump(model, "hotel_rating_model.pkl")

print("Model trained successfully!")
print("Total reviews:", len(data))
print("RMSE:", round(rmse, 4))
print("Model saved successfully!")