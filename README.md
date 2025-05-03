# Sentiment-analysis-using-NLP
---

**COMPANY:** CODTECH IT SOLUTIONS  
**NAME:** BHARKAVI P M  
**INTERN ID:** CT04DK897  
**DOMAIN:** MACHINE LEARNING  
**DURATION:** 4 WEEKS  (30th APR - 30th MAY)  
**MENTOR:** NEELA SANTHOSH  
**Current date:** Saturday, May 03, 2025, 5:45 PM IST

---

## **Project Overview**

This project demonstrates how to perform **Sentiment Analysis** on a dataset of customer reviews using **Natural Language Processing (NLP)** techniques. The workflow includes preprocessing the text data, transforming it into numerical features using **TF-IDF Vectorization**, and building a **Logistic Regression** model to classify sentiment. The deliverable is a **Jupyter Notebook** that showcases each step, from data cleaning to model evaluation and visualization.

---

## **Motivation**

Understanding customer sentiment is crucial for businesses to improve their products and services. **Sentiment Analysis** leverages NLP and machine learning to automatically determine whether a customer review is positive or negative. This project provides hands-on experience with essential NLP tasks and demonstrates how to build an end-to-end sentiment analysis pipeline using Python and scikit-learn.

---

## **Tools and Technologies**

- **Python:** Programming language for implementation.
- **Jupyter Notebook:** Platform for interactive coding, visualization, and documentation.
- **Pandas & NumPy:** Data manipulation and numerical operations.
- **Regular Expressions (re) & string:** Text preprocessing.
- **Matplotlib & Seaborn:** Data and model evaluation visualization.
- **scikit-learn:** Machine learning library for:
  - **TfidfVectorizer** for feature extraction.
  - **LogisticRegression** for classification.
  - **train_test_split, accuracy_score, classification_report, confusion_matrix** for model training and evaluation.

---

## **Dataset**

The dataset consists of customer reviews, each stored in a column named **'text'**. For demonstration purposes, mock sentiment labels (1 for positive, 0 for negative) are generated. In a real-world scenario, sentiment labels would be derived from actual review ratings or manual annotation.

---

## **Workflow**

1. **Import Libraries:**  
   Load all necessary libraries for data handling, preprocessing, modeling, and visualization.

2. **Load Dataset:**  
   Read the customer reviews dataset and confirm the presence of the required **'text'** column.

3. **Preprocess Text:**  
   Clean the text data by:
   - Lowercasing,
   - Removing punctuation, HTML tags, URLs, and numbers,
   - Stripping extra spaces and special characters.

4. **Label Sentiment:**  
   For demonstration, alternate sentiment labels (1 and 0) are assigned to the reviews.

5. **TF-IDF Vectorization:**  
   Convert cleaned text into numerical features using **TF-IDF**, capturing the importance of each word in the review.

6. **Train-Test Split:**  
   Divide the data into training and testing sets to ensure unbiased model evaluation.

7. **Model Training:**  
   Train a **Logistic Regression** model on the TF-IDF features.

8. **Prediction:**  
   Predict sentiment labels for the test set.

9. **Evaluation:**  
   Assess model performance using:
   - **Accuracy Score**
   - **Classification Report** (precision, recall, F1-score)
   - **Confusion Matrix**

10. **Visualization:**  
    Visualize the confusion matrix using a heatmap for better interpretability.

---

## **How to Run**

1. Ensure `customer_reviews.csv` is present in your working directory.
2. Open the provided Jupyter Notebook.
3. Execute each cell sequentially to reproduce the workflow, from preprocessing to evaluation and visualization.

---

## **Learning Outcomes**

- Gain practical experience in **NLP preprocessing** and **feature extraction**.
- Understand the application of **TF-IDF** and **Logistic Regression** for text classification.
- Learn to evaluate and visualize classification results.
- Build a foundational pipeline for more complex NLP projects.

---

## **Conclusion**

This project provides a comprehensive introduction to **Sentiment Analysis** using **NLP** and **machine learning**. By following the steps in the notebook, you will learn how to preprocess text, extract meaningful features, train a classifier, and interpret the results. The workflow is easily adaptable to real-world datasets and more advanced NLP tasks.

---
