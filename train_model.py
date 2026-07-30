import pandas as pd
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("emotions.csv")

# Model
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression())
])

# Train model
model.fit(df["text"], df["emotion"])

# Save model
pickle.dump(model, open("emotion_model.pkl", "wb"))

print("✅ Model Trained Successfully")