# import pandas as pd
# import numpy as np
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import nltk
# import re
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
import pickle


# nltk.download('stopwords')

# # load data
# data = pd.read_csv("data.csv")
# print(data.head())

# # data preprocessing
# label = LabelEncoder()
# y = label.fit_transform(data["Sentiment"])
# X = data["Sentence"]
# output = label.classes_

# print("Possible outputs:", output)

# ps = PorterStemmer()
# corpus = []
# for i in range(len(X)):
#     review = re.sub("[^a-zA-z]", " ", X[i])
#     review = review.lower()
#     review = review.split()
#     review = [ps.stem(word) for word in review if word not in set(
#         stopwords.words("english"))]
#     review = " ".join(review)
#     corpus.append(review)

# vectorizer = TfidfVectorizer()
# X_vectorized = vectorizer.fit_transform(corpus).toarray()

# # split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X_vectorized, y, test_size=0.2, random_state=101)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# # train model
# model = MultinomialNB()
# # model.fit(X_train, y_train)

# # test model
# y_pred = model.predict(X_test)
# acc = accuracy_score(y_pred, y_test)
# print("Accuracy of the model:", acc)

# # save model
# pickle.dump(model, open("model.pkl", "wb"))
# pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
# pickle.dump(output, open("output.pkl", "wb"))

# load model
ld_model = pickle.load(open("model.pkl", "rb"))
ld_vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
ld_output = pickle.load(open("output.pkl", "rb"))


# test model
def test_model(sentence):
    sen = ld_vectorizer.transform([sentence]).toarray()
    res = ld_model.predict(sen)
    return ld_output[res]


sen = "it gonna have a tremendous "
res = test_model(sen)
print(res)
