from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data:
df = pd.read_csv("diabetes.csv", sep=",")
print(df.head())
print("Column names:", df.columns)

# ploting the attributes vs output:
for i in df:
    #     print(df[i])
    plt.scatter(df[i], df.Outcome)
    plt.title(f"{i} vs Outcome")
    plt.xlabel(i)
    plt.ylabel("outcome")
    plt.show()

# splitting the output data
data = df.drop("Outcome", axis=1)
output = df["Outcome"]

# Feature Selection
selector = SelectKBest(score_func=chi2, k=5)  # Example using chi-square test
X_selected = selector.fit_transform(data, output)

# selected features
selected_indices = selector.get_support(indices=True)

print(data.columns[selected_indices])

# Scale the input data
scaler = StandardScaler()
data = scaler.fit_transform(X_selected)

# splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    data, output, test_size=0.2, random_state=42)

# creating the model

# Logistic regression:
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", accuracy)


# Random Forest:
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predicions:
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy:", accuracy)


# Naive Bayes:
# Create a Multinomial Naive Bayes classifier
model = GaussianNB()

# Train the classifier
model.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Naive Bayes Accuracy:", accuracy)


# SVM
# Scale the features using StandardScaler
model = SVC(kernel='linear')

# Train the classifier
model.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("SVM Accuracy:", accuracy)
