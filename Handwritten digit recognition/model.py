from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


# # preapring traing and validation dataset.
# train = ImageDataGenerator(rescale=1/255)
# validation = ImageDataGenerator(rescale=1/255)

# train_dataset = train.flow_from_directory(
#     "archive/trainingSet/trainingSet", target_size=(28, 28), batch_size=3, class_mode="categorical")
# validation_dataset = validation.flow_from_directory(
#     "archive/trainingSample/trainingSample", target_size=(28, 28), batch_size=3, class_mode="categorical")

# print(train_dataset.class_indices)
# print(train_dataset[0][0].shape)


# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28, 3)),
#     tf.keras.layers.Dense(128, activation="relu"),
#     tf.keras.layers.Dense(128, activation="relu"),
#     tf.keras.layers.Dense(10, activation="softmax")
# ])


# model.compile(optimizer="adam", loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(train_dataset, steps_per_epoch=50, epochs=10,
#           validation_data=validation_dataset)

# model.save("handwritten_digit.model")

model = tf.keras.models.load_model("handwritten_digit.model")
image_number = 0
while os.path.isfile(f"data/img_{image_number}.jpg"):
    img = cv2.imread(f"data/img_{image_number}.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB format
    img = cv2.resize(img, (28, 28))  # Resize image to (28, 28)
    img = img / 255.0  # Normalize pixel values between 0 and 1
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    print(f"This digit is probably a {predicted_label}")

    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()

    image_number += 1
