#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Step 1: Import Libraries
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[2]:


# Step 2: Load Dataset
df = pd.read_csv("customer_reviews.csv")  # Replace with your actual file name
print("Initial Columns:", df.columns)

# Confirming correct column name
if 'text' not in df.columns:
    raise ValueError("Expected a column named 'text' in your dataset.")


# In[3]:


# Step 3: Preprocess Text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

# Confirm preprocessing step
print("\nSample cleaned text:")
print(df['cleaned_text'].head())


# In[4]:


# Step 4: Add Mock Sentiment Labels for Testing
df['Sentiment'] = [1 if i % 2 == 0 else 0 for i in range(len(df))]  # Alternate 1 and 0


# In[5]:


# Step 5: TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_text']).toarray()
y = df['Sentiment']


# In[6]:


# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# Step 7: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)


# In[8]:


# Step 8: Predictions
y_pred = model.predict(X_test)


# In[9]:


# Step 9: Evaluation
print("\n✅ Accuracy Score:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧾 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[10]:


# Step 10: Visualization
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

