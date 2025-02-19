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
# The regularization techniques mentioned in chapter 5 are L1, L2, and L1+L2 Weight Regularization.
# Other names for these techniques are Lasso, Ridge, and Elastic Net, respectively.

# %% colab_type="code"
from six import u
from tensorflow.keras.datasets import mnist
import numpy as np
import random

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

print(train_labels.shape)

filtered_train_images = [train_images[train_labels == i] for i in range(10)]
filtered_test_images  = [test_images[(test_labels == i)] for i in range(10)]

# %% colab_type="code"
num_train_pairs = 0
num_test_pairs = 0

for i in range(0, 5):
    j = i * 2
    random.shuffle(filtered_train_images[j])
    random.shuffle(filtered_train_images[j+1])

    random.shuffle(filtered_test_images[j])
    random.shuffle(filtered_test_images[j+1])

    num_train_pairs += (
        min(len(filtered_train_images[j]), len(filtered_train_images[j+1]))
    )

    num_test_pairs += (
        min(len(filtered_test_images[j]), len(filtered_test_images[j+1]))
    )

print(f"num_train_pairs = {num_train_pairs}")
print(f"num_test_pairs = {num_test_pairs}")

# %% colab_type="code"
train_images = np.zeros((num_train_pairs, 28 * 28))
train_labels = np.zeros((num_train_pairs))

test_images = np.zeros((num_test_pairs, 28 * 28))
test_labels = np.zeros((num_test_pairs))

print(filtered_train_images[0][0].shape)

# %% colab_type="code"
train_idx = 0
test_idx = 0
for i in range(0, 5):
    j = i * 2

    min_train_len = (
        min(len(filtered_train_images[j]), len(filtered_train_images[j+1]))
    )

    for x in range(min_train_len):
        train_images[train_idx, :] = (
            np.mean([filtered_train_images[j][x], filtered_train_images[j+1][x]], axis=0)
        )
        train_labels[train_idx] = i
        train_idx += 1

    min_test_len = (
        min(len(filtered_test_images[j]), len(filtered_test_images[j+1]))
    )

    for x in range(min_test_len):
        test_images[test_idx, :] = (
            np.mean([filtered_test_images[j][x], filtered_test_images[j+1][x]], axis=0)
        )
        test_labels[test_idx] = i
        test_idx += 1

# %% colab_type="code"
import matplotlib.pyplot as plt

# %% colab_type="code"
fig, axes = plt.subplots(5, 2, figsize=(10, 15))

for i in range(5):
    reshaped_0 = train_images[train_labels==i][0].reshape(28, 28)
    axes[i][0].imshow(reshaped_0, cmap='gray')
    axes[i][0].axis('off')  # Turn off axis labels

    reshaped_1 = train_images[train_labels==i][1].reshape(28, 28)
    axes[i][1].imshow(reshaped_1, cmap='gray')
    axes[i][1].axis('off')  # Turn off axis labels

plt.tight_layout()
plt.show()

# %% [markdown] colab_type="text"
# Question 3

# %% colab_type="code"
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
import keras_tuner as kt

# %% colab_type="code"
def build_model(hp):
    # Define hyperparameters
    units = hp.Int('units', min_value=256, max_value=512, step=128)
    optimizer = hp.Choice('optimizer', ['rmsprop', 'adam', 'sgd'])
    learning_rate = hp.Choice('learning_rate', [0.001, 0.005, 0.01])

    # Define optimizer with learning rate
    if optimizer == "rmsprop":
        opt = optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == "adam":
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "sgd":
        opt = optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    model = Sequential()

    regularization_type = hp.Choice('regularization', values=['l1', 'l2', 'l1_l2', 'dropout', 'none'])
    l1_reg = hp.Float('l1_reg', min_value=1e-5, max_value=1e-4, step=1e-5)
    l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-4, step=1e-5)
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.2, step=0.05)
    
    if regularization_type == 'l1':
        model.add(Dense(units, activation='relu', 
                kernel_regularizer=regularizers.l1(l1_reg)))
    
    elif regularization_type == 'l2':
        model.add(Dense(units, activation='relu', 
                kernel_regularizer=regularizers.l2(l2_reg)))
    
    elif regularization_type == 'l1_l2':
        model.add(Dense(units, activation='relu', 
                kernel_regularizer=regularizers.l1_l2(l1=l1_reg, l2=l2_reg)))

    elif regularization_type == 'dropout' or regularization_type == 'none':
        model.add(Dense(units, activation='relu'))
        if regularization_type == 'dropout':
            model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(10, activation='softmax'))

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# %% colab_type="code"
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=15,
    executions_per_trial=1,
    directory='my_tuner_dir',
    project_name='homework3'
)

# %% colab_type="code"
tuner.search(train_images, train_labels, validation_split=0.2)

# %% colab_type="code"
best_trial = tuner.oracle.get_best_trials()[0]
best_hps   = best_trial.hyperparameters

print("Best Hyperparameters:")
for hp, val in best_hps.values.items():
    print(f"{hp}: {val}")

# %% colab_type="code"
model = tuner.hypermodel.build(best_hps)
history = model.fit(train_images, train_labels, epochs=15, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
# best_epoch = 5
print('Best epoch: %d' % (best_epoch,))

# %% colab_type="code"
hypermodel = tuner.hypermodel.build(best_hps)
history = hypermodel.fit(train_images, train_labels, epochs=best_epoch, validation_split=0.2)

eval_result = hypermodel.evaluate(test_images, test_labels)
print("[test loss, test accuracy]:", eval_result)
