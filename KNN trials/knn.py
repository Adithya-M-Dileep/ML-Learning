import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Read data from csv file
data = pd.read_csv("car.data", sep=",")

print(data.head())

# Convert data to numerical values
le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

# Create features and labels
predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)


# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Create model
best_acc = 0
best_k = 0
model = None
for i in range(1, 12):
    temp_model = KNeighborsClassifier(n_neighbors=i)
    temp_model.fit(x_train, y_train)
    acc = temp_model.score(x_test, y_test)
    if acc > best_acc:
        model = temp_model
        best_acc = acc
        best_k = i

print(best_acc, "neighbours:", best_k)

# Make predictions
predictions = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]
# Print predictions
for i in range(len(predictions)):
    print("Predicted:", names[predictions[i]],
          "Data:", x_test[i], "Actual:", names[y_test[i]])
    n = model.kneighbors([x_test[i]], 9, True)
    print("N:", n)
