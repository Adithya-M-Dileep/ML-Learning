import pandas as pa
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as pyplot
from matplotlib import style

# Read the data from the file
data = pa.read_csv("student-mat.csv", sep=";")
print(data.head())

# Select the columns that we want to use
data = data[["G1", "G2", "G3", "health", "absences", "studytime"]]

print(data.head())
predict = "G3"

X = np.array(data.drop(predict, axis="columns"))
y = np.array(data[predict])

# Linear regression

linear = linear_model.LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

best = 0
# for _ in range(30):
#     x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
#     linear.fit(x_train, y_train)
#     acc = linear.score(x_test, y_test)
#     print(acc)

#     if acc > best:
#         best = acc
#         with open("studetModle.pickle", "wb") as f:
#             pickle.dump(linear, f)

pickle_in = open("studetModle.pickle", "rb")
linear = pickle.load(pickle_in)
print("Coefficient: ", linear.coef_)
print("Intersept:", linear.intercept_)

prediction = linear.predict(x_test)

for x in range(len(prediction)):
    print(x_test[x], ",", y_test[x], ":", prediction[x])


style.use("ggplot")
pyplot.scatter(data["G1"], data["G3"])
pyplot.xlabel("G1")
pyplot.ylabel("G3")
pyplot.show()
