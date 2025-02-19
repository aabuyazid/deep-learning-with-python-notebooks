import tensorflow as tf 

class NaiveDense:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation

        w_shape = (input_size, output_size)
        w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
        self.W = tf.Variable(w_initial_value)

        b_shape = (output_size,)
        b_initial_value = tf.zeros(b_shape)
        self.b = tf.Variable(b_initial_value)

    def __call__(self, inputs):
        return self.activation(tf.matmul(inputs, self.W) + self.b)

    @property
    def weights(self):
        return [self.W, self.b]


# %% [markdown] colab_type="text"
# #### A simple Sequential class

# %% colab_type="code"
class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
           x = layer(x)

        return x

    @property
    def weights(self):
       weights = []
       for layer in self.layers:
           weights += layer.weights
       return weights




# %% [markdown] colab_type="text"
# #### A batch generator

# %% colab_type="code"
import math

class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)

    def next(self):
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels


def fit(model, optimizer, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter}")
        batch_generator = BatchGenerator(images, labels)
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, optimizer, images_batch, labels_batch)
            if batch_counter % 100 == 0:
                print(f"loss at batch {batch_counter}: {loss:.2f}")

# %% colab_type="code"
def one_training_step(model, optimizer, images_batch, labels_batch):
    with tf.GradientTape() as tape:
        predictions = model(images_batch)
        per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(
            labels_batch, predictions)
        average_loss = tf.reduce_mean(per_sample_losses)
        gradients = tape.gradient(average_loss, model.weights)
        update_weights(optimizer, gradients, model.weights)
        return average_loss


# %% colab_type="code"
learning_rate = 1e-3

# %% colab_type="code"
from tensorflow.keras import optimizers

optimizer = optimizers.SGD(learning_rate=1e-3)

def update_weights(optimizer, gradients, weights):
    optimizer.apply_gradients(zip(gradients, weights))


# Start of my Code
from keras.datasets import mnist
import matplotlib.pyplot as plt, numpy as np

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

    optimizer = optimizers.SGD(learning_rate=1e-3)
    model = NaiveSequential(layers)

    fit(model, optimizer, train_images, train_labels, epochs=10, batch_size=128)
    
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
