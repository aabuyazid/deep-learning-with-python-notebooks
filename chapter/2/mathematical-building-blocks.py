# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] colab_type="text"
# This is a companion notebook for the book [Deep Learning with Python, Second Edition](https://www.manning.com/books/deep-learning-with-python-second-edition?a_aid=keras&a_bid=76564dff). For readability, it only contains runnable code blocks and section titles, and omits everything else in the book: text paragraphs, figures, and pseudocode.
#
# **If you want to be able to follow what's going on, I recommend reading the notebook side by side with your copy of the book.**
#
# This notebook was generated for TensorFlow 2.6.

# %% [markdown] colab_type="text"
# # The mathematical building blocks of neural networks

# %% [markdown] colab_type="text"
# ## A first look at a neural network

# %% [markdown] colab_type="text"
# **Loading the MNIST dataset in Keras**

# %% colab_type="code"
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# %% colab_type="code"
train_images.shape

# %% colab_type="code"
len(train_labels)

# %% colab_type="code"
train_labels

# %% colab_type="code"
test_images.shape

# %% colab_type="code"
len(test_labels)

# %% colab_type="code"
test_labels

# %% [markdown] colab_type="text"
# **The network architecture**

# %% colab_type="code"
from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# %% [markdown] colab_type="text"
# **The compilation step**

# %% colab_type="code"
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# %% [markdown] colab_type="text"
# **Preparing the image data**

# %% colab_type="code"
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# %% [markdown] colab_type="text"
# **"Fitting" the model**

# %% colab_type="code"
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# %% [markdown] colab_type="text"
# **Using the model to make predictions**

# %% colab_type="code"
test_digits = test_images[0:10]
predictions = model.predict(test_digits)
predictions[0]

# %% colab_type="code"
predictions[0].argmax()

# %% colab_type="code"
predictions[0][7]

# %% colab_type="code"
test_labels[0]

# %% [markdown] colab_type="text"
# **Evaluating the model on new data**

# %% colab_type="code"
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")

# %% [markdown] colab_type="text"
# ## Data representations for neural networks

# %% [markdown] colab_type="text"
# ### Scalars (rank-0 tensors)

# %% colab_type="code"
import numpy as np
x = np.array(12)
x

# %% colab_type="code"
x.ndim

# %% [markdown] colab_type="text"
# ### Vectors (rank-1 tensors)

# %% colab_type="code"
x = np.array([12, 3, 6, 14, 7])
x

# %% colab_type="code"
x.ndim

# %% [markdown] colab_type="text"
# ### Matrices (rank-2 tensors)

# %% colab_type="code"
x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]])
x.ndim

# %% [markdown] colab_type="text"
# ### Rank-3 and higher-rank tensors

# %% colab_type="code"
x = np.array([[[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [6, 79, 3, 35, 1],
               [7, 80, 4, 36, 2]]])
x.ndim

# %% [markdown] colab_type="text"
# ### Key attributes

# %% colab_type="code"
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# %% colab_type="code"
train_images.ndim

# %% colab_type="code"
train_images.shape

# %% colab_type="code"
train_images.dtype

# %% [markdown] colab_type="text"
# **Displaying the fourth digit**

# %% colab_type="code"
import matplotlib.pyplot as plt
digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# %% colab_type="code"
train_labels[4]

# %% [markdown] colab_type="text"
# ### Manipulating tensors in NumPy

# %% colab_type="code"
my_slice = train_images[10:100]
my_slice.shape

# %% colab_type="code"
my_slice = train_images[10:100, :, :]
my_slice.shape

# %% colab_type="code"
my_slice = train_images[10:100, 0:28, 0:28]
my_slice.shape

# %% colab_type="code"
my_slice = train_images[:, 14:, 14:]

# %% colab_type="code"
my_slice = train_images[:, 7:-7, 7:-7]

# %% [markdown] colab_type="text"
# ### The notion of data batches

# %% colab_type="code"
batch = train_images[:128]

# %% colab_type="code"
batch = train_images[128:256]

# %% colab_type="code"
n = 3
batch = train_images[128 * n:128 * (n + 1)]


# %% [markdown] colab_type="text"
# ### Real-world examples of data tensors

# %% [markdown] colab_type="text"
# ### Vector data

# %% [markdown] colab_type="text"
# ### Timeseries data or sequence data

# %% [markdown] colab_type="text"
# ### Image data

# %% [markdown] colab_type="text"
# ### Video data

# %% [markdown] colab_type="text"
# ## The gears of neural networks: tensor operations

# %% [markdown] colab_type="text"
# ### Element-wise operations

# %% colab_type="code"
def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x


# %% colab_type="code"
def naive_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x


# %% colab_type="code"
import time

x = np.random.random((20, 100))
y = np.random.random((20, 100))

t0 = time.time()
for _ in range(1000):
    z = x + y
    z = np.maximum(z, 0.)
print("Took: {0:.2f} s".format(time.time() - t0))

# %% colab_type="code"
t0 = time.time()
for _ in range(1000):
    z = naive_add(x, y)
    z = naive_relu(z)
print("Took: {0:.2f} s".format(time.time() - t0))

# %% [markdown] colab_type="text"
# ### Broadcasting

# %% colab_type="code"
import numpy as np
X = np.random.random((32, 10))
y = np.random.random((10,))

# %% colab_type="code"
y = np.expand_dims(y, axis=0)

# %% colab_type="code"
Y = np.concatenate([y] * 32, axis=0)


# %% colab_type="code"
def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x


# %% colab_type="code"
import numpy as np
x = np.random.random((64, 3, 32, 10))
y = np.random.random((32, 10))
z = np.maximum(x, y)

# %% [markdown] colab_type="text"
# ### Tensor product

# %% colab_type="code"
x = np.random.random((32,))
y = np.random.random((32,))
z = np.dot(x, y)


# %% colab_type="code"
def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z


# %% colab_type="code"
def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z


# %% colab_type="code"
def naive_matrix_vector_dot(x, y):
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = naive_vector_dot(x[i, :], y)
    return z


# %% colab_type="code"
def naive_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]
    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)
    return z


# %% [markdown] colab_type="text"
# ### Tensor reshaping

# %% colab_type="code"
train_images = train_images.reshape((60000, 28 * 28))

# %% colab_type="code"
x = np.array([[0., 1.],
             [2., 3.],
             [4., 5.]])
x.shape

# %% colab_type="code"
x = x.reshape((6, 1))
x

# %% colab_type="code"
x = np.zeros((300, 20))
x = np.transpose(x)
x.shape

# %% [markdown] colab_type="text"
# ### Geometric interpretation of tensor operations

# %% [markdown] colab_type="text"
# ### A geometric interpretation of deep learning

# %% [markdown] colab_type="text"
# ## The engine of neural networks: gradient-based optimization

# %% [markdown] colab_type="text"
# ### What's a derivative?

# %% [markdown] colab_type="text"
# ### Derivative of a tensor operation: the gradient

# %% [markdown] colab_type="text"
# ### Stochastic gradient descent

# %% [markdown] colab_type="text"
# ### Chaining derivatives: The Backpropagation algorithm

# %% [markdown] colab_type="text"
# #### The chain rule

# %% [markdown] colab_type="text"
# #### Automatic differentiation with computation graphs

# %% [markdown] colab_type="text"
# #### The gradient tape in TensorFlow

# %% colab_type="code"
import tensorflow as tf
x = tf.Variable(0.)
with tf.GradientTape() as tape:
    y = 2 * x + 3
grad_of_y_wrt_x = tape.gradient(y, x)

# %% colab_type="code"
x = tf.Variable(tf.random.uniform((2, 2)))
with tf.GradientTape() as tape:
    y = 2 * x + 3
grad_of_y_wrt_x = tape.gradient(y, x)

# %% colab_type="code"
W = tf.Variable(tf.random.uniform((2, 2)))
b = tf.Variable(tf.zeros((2,)))
x = tf.random.uniform((2, 2))
with tf.GradientTape() as tape:
    y = tf.matmul(x, W) + b
grad_of_y_wrt_W_and_b = tape.gradient(y, [W, b])

# %% [markdown] colab_type="text"
# ## Looking back at our first example

# %% colab_type="code"
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# %% colab_type="code"
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# %% colab_type="code"
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# %% colab_type="code"
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# %% [markdown] colab_type="text"
# ### Reimplementing our first example from scratch in TensorFlow

# %% [markdown] colab_type="text"
# #### A simple Dense class

# %% colab_type="code"
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


# %% colab_type="code"
model = NaiveSequential([
    NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
    NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)
])
assert len(model.weights) == 4

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


# %% [markdown] colab_type="text"
# ### Running one training step

# %% colab_type="code"
def one_training_step(model, images_batch, labels_batch):
    with tf.GradientTape() as tape:
        predictions = model(images_batch)
        per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(
            labels_batch, predictions)
        average_loss = tf.reduce_mean(per_sample_losses)
    gradients = tape.gradient(average_loss, model.weights)
    update_weights(gradients, model.weights)
    return average_loss


# %% colab_type="code"
learning_rate = 1e-3

def update_weights(gradients, weights):
    for g, w in zip(gradients, weights):
        w.assign_sub(g * learning_rate)


# %% colab_type="code"
from tensorflow.keras import optimizers

optimizer = optimizers.SGD(learning_rate=1e-3)

def update_weights(gradients, weights):
    optimizer.apply_gradients(zip(gradients, weights))


# %% [markdown] colab_type="text"
# ### The full training loop

# %% colab_type="code"
def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter}")
        batch_generator = BatchGenerator(images, labels)
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = one_training_step(model, images_batch, labels_batch)
            if batch_counter % 100 == 0:
                print(f"loss at batch {batch_counter}: {loss:.2f}")


# %% colab_type="code"
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

fit(model, train_images, train_labels, epochs=10, batch_size=128)

# %% [markdown] colab_type="text"
# ### Evaluating the model

# %% colab_type="code"
predictions = model(test_images)
predictions = predictions.numpy()
predicted_labels = np.argmax(predictions, axis=1)
matches = predicted_labels == test_labels
print(f"accuracy: {matches.mean():.2f}")

# %% [markdown] colab_type="text"
# ## Summary
