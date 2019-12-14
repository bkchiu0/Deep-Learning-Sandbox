import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

from prediction_plotter import plot_image, plot_value_array

data = datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

# Normalize the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Add in
model = models.Sequential([
    layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    layers.Conv2D(128, (3, 3), activation='relu',
                  input_shape=(28, 28, 1), use_bias=True),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', use_bias=True),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', use_bias=True),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

print(model.summary())

model.compile(optimizer="nadam", loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5,
          validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# Visualize predictions

predictions = model.predict(test_images)
num_rows = 5
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
