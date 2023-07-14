import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data from CSV file
raw_mail_data = pd.read_csv('CUsersDELLDownloadsspammail_data.csv')

# Replace null values with a null string
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

# Label encoding
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

# Separating the data into texts and labels
X = mail_data['Message']
Y = mail_data['Category']

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Feature extraction using TF-IDF
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)

# Convert labels to integers
Y_train = Y_train.astype('int')

# Logistic Regression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train_features, Y_train)

# Naive Bayes
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_features, Y_train)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train_features, Y_train)

def classify_mail(mail_content)
    input_data_features = feature_extraction.transform([mail_content])
    prediction_logreg = logistic_regression.predict(input_data_features)[0]
    prediction_nb = naive_bayes.predict(input_data_features)[0]
    prediction_rf = random_forest.predict(input_data_features)[0]
    return prediction_logreg, prediction_nb, prediction_rf

def get_accuracy(predictions, ground_truth)
    return accuracy_score(ground_truth, predictions)

# Example usage
input_mail = Hello, this is a spam message.
logreg_prediction, nb_prediction, rf_prediction = classify_mail(input_mail)

print(Logistic Regression Prediction, logreg_prediction)
print(Naive Bayes Prediction, nb_prediction)
print(Random Forest Prediction, rf_prediction)

# Calculate accuracies
logreg_accuracy = get_accuracy(logistic_regression.predict(X_train_features), Y_train)
nb_accuracy = get_accuracy(naive_bayes.predict(X_train_features), Y_train)
rf_accuracy = get_accuracy(random_forest.predict(X_train_features), Y_train)

print(Logistic Regression Accuracy, logreg_accuracy)
print(Naive Bayes Accuracy, nb_accuracy)
print(Random Forest Accuracy, rf_accuracy)
