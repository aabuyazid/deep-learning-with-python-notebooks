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
# # Getting started with neural networks: Classification and regression

# %% [markdown] colab_type="text"
# ## Classifying movie reviews: A binary classification example

# %% [markdown] colab_type="text"
# ### The IMDB dataset

# %% [markdown] colab_type="text"
# **Loading the IMDB dataset**

# %% colab_type="code"
from tensorflow.keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)

# %% colab_type="code"
train_data[0]

# %% colab_type="code"
train_labels[0]

# %% colab_type="code"
max([max(sequence) for sequence in train_data])

# %% [markdown] colab_type="text"
# **Decoding reviews back to text**

# %% colab_type="code"
word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])
decoded_review = " ".join(
    [reverse_word_index.get(i - 3, "?") for i in train_data[0]])

# %% [markdown] colab_type="text"
# ### Preparing the data

# %% [markdown] colab_type="text"
# **Encoding the integer sequences via multi-hot encoding**

# %% colab_type="code"
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# %% colab_type="code"
x_train[0]

# %% colab_type="code"
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

# %% [markdown] colab_type="text"
# ### Building your model

# %% [markdown] colab_type="text"
# **Model definition**

# %% colab_type="code"
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# %% [markdown] colab_type="text"
# **Compiling the model**

# %% colab_type="code"
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# %% [markdown] colab_type="text"
# ### Validating your approach

# %% [markdown] colab_type="text"
# **Setting aside a validation set**

# %% colab_type="code"
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# %% [markdown] colab_type="text"
# **Training your model**

# %% colab_type="code"
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# %% colab_type="code"
history_dict = history.history
history_dict.keys()

# %% [markdown] colab_type="text"
# **Plotting the training and validation loss**

# %% colab_type="code"
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %% [markdown] colab_type="text"
# **Plotting the training and validation accuracy**

# %% colab_type="code"
plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# %% [markdown] colab_type="text"
# **Retraining a model from scratch**

# %% colab_type="code"
model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

# %% colab_type="code"
results

# %% [markdown] colab_type="text"
# ### Using a trained model to generate predictions on new data

# %% colab_type="code"
model.predict(x_test)

# %% [markdown] colab_type="text"
# ### Further experiments

# %% [markdown] colab_type="text"
# ### Wrapping up

# %% [markdown] colab_type="text"
# ## Classifying newswires: A multiclass classification example

# %% [markdown] colab_type="text"
# ### The Reuters dataset

# %% [markdown] colab_type="text"
# **Loading the Reuters dataset**

# %% colab_type="code"
from tensorflow.keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    num_words=10000)

# %% colab_type="code"
len(train_data)

# %% colab_type="code"
len(test_data)

# %% colab_type="code"
train_data[10]

# %% [markdown] colab_type="text"
# **Decoding newswires back to text**

# %% colab_type="code"
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = " ".join([reverse_word_index.get(i - 3, "?") for i in
    train_data[0]])

# %% colab_type="code"
train_labels[10]

# %% [markdown] colab_type="text"
# ### Preparing the data

# %% [markdown] colab_type="text"
# **Encoding the input data**

# %% colab_type="code"
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


# %% [markdown] colab_type="text"
# **Encoding the labels**

# %% colab_type="code"
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results
y_train = to_one_hot(train_labels)
y_test = to_one_hot(test_labels)

# %% colab_type="code"
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

# %% [markdown] colab_type="text"
# ### Building your model

# %% [markdown] colab_type="text"
# **Model definition**

# %% colab_type="code"
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(46, activation="softmax")
])

# %% [markdown] colab_type="text"
# **Compiling the model**

# %% colab_type="code"
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# %% [markdown] colab_type="text"
# ### Validating your approach

# %% [markdown] colab_type="text"
# **Setting aside a validation set**

# %% colab_type="code"
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

# %% [markdown] colab_type="text"
# **Training the model**

# %% colab_type="code"
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# %% [markdown] colab_type="text"
# **Plotting the training and validation loss**

# %% colab_type="code"
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %% [markdown] colab_type="text"
# **Plotting the training and validation accuracy**

# %% colab_type="code"
plt.clf()
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# %% [markdown] colab_type="text"
# **Retraining a model from scratch**

# %% colab_type="code"
model = keras.Sequential([
  layers.Dense(64, activation="relu"),
  layers.Dense(64, activation="relu"),
  layers.Dense(46, activation="softmax")
])
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_train,
          y_train,
          epochs=9,
          batch_size=512)
results = model.evaluate(x_test, y_test)

# %% colab_type="code"
results

# %% colab_type="code"
import copy
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
hits_array.mean()

# %% [markdown] colab_type="text"
# ### Generating predictions on new data

# %% colab_type="code"
predictions = model.predict(x_test)

# %% colab_type="code"
predictions[0].shape

# %% colab_type="code"
np.sum(predictions[0])

# %% colab_type="code"
np.argmax(predictions[0])

# %% [markdown] colab_type="text"
# ### A different way to handle the labels and the loss

# %% colab_type="code"
y_train = np.array(train_labels)
y_test = np.array(test_labels)

# %% colab_type="code"
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# %% [markdown] colab_type="text"
# ### The importance of having sufficiently large intermediate layers

# %% [markdown] colab_type="text"
# **A model with an information bottleneck**

# %% colab_type="code"
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(4, activation="relu"),
    layers.Dense(46, activation="softmax")
])
model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.fit(partial_x_train,
          partial_y_train,
          epochs=20,
          batch_size=128,
          validation_data=(x_val, y_val))

# %% [markdown] colab_type="text"
# ### Further experiments

# %% [markdown] colab_type="text"
# ### Wrapping up

# %% [markdown] colab_type="text"
# ## Predicting house prices: A regression example

# %% [markdown] colab_type="text"
# ### The Boston Housing Price dataset

# %% [markdown] colab_type="text"
# **Loading the Boston housing dataset**

# %% colab_type="code"
from tensorflow.keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# %% colab_type="code"
train_data.shape

# %% colab_type="code"
test_data.shape

# %% colab_type="code"
train_targets

# %% [markdown] colab_type="text"
# ### Preparing the data

# %% [markdown] colab_type="text"
# **Normalizing the data**

# %% colab_type="code"
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std


# %% [markdown] colab_type="text"
# ### Building your model

# %% [markdown] colab_type="text"
# **Model definition**

# %% colab_type="code"
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model


# %% [markdown] colab_type="text"
# ### Validating your approach using K-fold validation

# %% [markdown] colab_type="text"
# **K-fold validation**

# %% colab_type="code"
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=16, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

# %% colab_type="code"
all_scores

# %% colab_type="code"
np.mean(all_scores)

# %% [markdown] colab_type="text"
# **Saving the validation logs at each fold**

# %% colab_type="code"
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=16, verbose=0)
    mae_history = history.history["val_mae"]
    all_mae_histories.append(mae_history)

# %% [markdown] colab_type="text"
# **Building the history of successive mean K-fold validation scores**

# %% colab_type="code"
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# %% [markdown] colab_type="text"
# **Plotting validation scores**

# %% colab_type="code"
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()

# %% [markdown] colab_type="text"
# **Plotting validation scores, excluding the first 10 data points**

# %% colab_type="code"
truncated_mae_history = average_mae_history[10:]
plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()

# %% [markdown] colab_type="text"
# **Training the final model**

# %% colab_type="code"
model = build_model()
model.fit(train_data, train_targets,
          epochs=130, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

# %% colab_type="code"
test_mae_score

# %% [markdown] colab_type="text"
# ### Generating predictions on new data

# %% colab_type="code"
predictions = model.predict(test_data)
predictions[0]

# %% [markdown] colab_type="text"
# ### Wrapping up

# %% [markdown] colab_type="text"
# ## Summary
