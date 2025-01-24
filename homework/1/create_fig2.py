import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt, numpy as np

from naive_sequential import NaiveDense, NaiveSequential, fit

num_layers  = [  2,  3,  4,  5]
num_neurons = [512, 10, 32, 16]
accuracy = []

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

for i, nl in enumerate(num_layers):
    prev_nn = num_neurons[0]
    layers = [
        NaiveDense(input_size=28 * 28, output_size=prev_nn, activation=tf.nn.relu)
    ]

    for nn in num_neurons[1:i+1]:
        print(f"nn = {nn}")
        layers.append(
            NaiveDense(input_size=prev_nn, output_size=nn, activation=tf.nn.relu)
        )
        prev_nn = nn
     
    layers.append(
        NaiveDense(input_size=prev_nn, output_size=10, activation=tf.nn.softmax)
    )

    assert len(layers) == nl, f"Expected {nl} layers, got {len(layers)}"

    model = NaiveSequential(layers)

    fit(model, train_images, train_labels, epochs=10, batch_size=128)
    
    # %% colab_type="code"
    predictions = model(test_images)
    predictions = predictions.numpy()
    predicted_labels = np.argmax(predictions, axis=1)
    matches = predicted_labels == test_labels
    accuracy.append(matches.mean())
    print(f"accuracy: {accuracy[-1]:.2f}")

    predictions = model(train_images)
    predictions = predictions.numpy()
    predicted_labels = np.argmax(predictions, axis=1)
    matches = predicted_labels == train_labels
    print(f"train accuracy: {matches.mean():.2f}\n")

    del model

# Create the plot
plt.figure(figsize=(8, 6))  # Set the figure size
plt.plot(num_layers, accuracy, marker='o', linestyle='-', color='b')

# Adding title and labels
plt.title("Number of Layers vs Accuracy")
plt.xlabel("# of Layers")  # x-axis label with units
plt.ylabel("Accuracy")  # y-axis label

# Show the plot
plt.grid(True)
plt.show()
