from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Assuming you have your texts and tags
texts = ["I love this movie", "The food was terrible", "What a fantastic book"]
tags = [["positive", "movie"], ["negative", "food"], ["positive", "book"]]

# Convert tags to binary vectors
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(tags)

# Build the text-to-tag classifier
clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", OneVsRestClassifier(RandomForestClassifier()))
])
clf.fit(texts, y)

# Predict tags for a new text
new_text = "This book is terrible"
predicted_tags = clf.predict([new_text])
predicted_tags_labels = mlb.inverse_transform(predicted_tags)  # convert binary vectors back to tag names
print(predicted_tags_labels)