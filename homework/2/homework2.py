from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import keras_tuner as kt
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb

# %% colab_type="code"
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)

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
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results
X_train = vectorize_sequences(train_data)
X_test = vectorize_sequences(test_data)

# %% colab_type="code"
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")# %% colab_type="code"

X_val = X_train[:10000]
partial_X_train = X_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

def build_model(hp):
    # Define hyperparameters
    units1 = hp.Int('units1', min_value=16, max_value=32, step=16)
    units2 = hp.Int('units2', min_value=16, max_value=32, step=16)
    optimizer = hp.Choice('optimizer', ['rmsprop', 'adam'])
    learning_rate = hp.Choice('learning_rate', [0.001, 0.01])

    # Define optimizer with learning rate
    if optimizer == "rmsprop":
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == "adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    # Build model
    model = keras.Sequential([
        layers.Dense(units1, activation="relu"),
        layers.Dense(units2, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Initialize the tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='my_tuner_dir',
    project_name='homework2'
)

print("Searching")
tuner.search(partial_X_train, partial_y_train, epochs=5, validation_split=0.2)

# After the search is complete, get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Print the best hyperparameters
print("Best Hyperparameters:")
print(f"units1: {best_hps.get('units1')}")
print(f"units2: {best_hps.get('units2')}")
print(f"optimizer: {best_hps.get('optimizer')}")
print(f"learning_rate: {best_hps.get('learning_rate')}")

model = tuner.hypermodel.build(best_hps)
history = model.fit(partial_X_train, partial_y_train, epochs=5, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
# best_epoch = 5
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)
history = hypermodel.fit(partial_X_train, partial_y_train, epochs=best_epoch, validation_split=0.2)

eval_result = hypermodel.evaluate(X_val, y_val)
print("[test loss, test accuracy]:", eval_result)
