import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt, numpy as np

from naive_sequential import NaiveDense, NaiveSequential, fit

size_first = [16, 32, 64, 128, 256, 512]
accuracy = []

for s in size_first:
    model = NaiveSequential([
        NaiveDense(input_size=28 * 28, output_size=s, activation=tf.nn.relu),
        NaiveDense(input_size=s, output_size=10, activation=tf.nn.softmax)
    ])
    assert len(model.weights) == 4
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype("float32") / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype("float32") / 255
    
    fit(model, train_images, train_labels, epochs=10, batch_size=128)
    
    # %% colab_type="code"
    predictions = model(test_images)
    predictions = predictions.numpy()
    predicted_labels = np.argmax(predictions, axis=1)
    matches = predicted_labels == test_labels
    accuracy.append(matches.mean())
    print(f"accuracy: {accuracy[-1]:.2f}\n")

    predictions = model(train_images)
    predictions = predictions.numpy()
    predicted_labels = np.argmax(predictions, axis=1)
    matches = predicted_labels == test_labels
    print(f"train accuracy: {accuracy[-1]:.2f}\n")

# Create the plot
plt.figure(figsize=(8, 6))  # Set the figure size
plt.plot(size_first, accuracy, marker='o', linestyle='-', color='b')

# Adding title and labels
plt.title("Size of First Layer vs Accuracy")
plt.xlabel("Size of First Layer (Number of Neurons)")  # x-axis label with units
plt.ylabel("Accuracy")  # y-axis label

# Show the plot
plt.grid(True)
plt.show()
