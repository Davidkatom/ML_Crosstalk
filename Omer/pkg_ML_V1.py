import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import clear_output
import os
import re

def super_plot(xl='x', yl='f(x)', title='', base_size=16, xtickx_size=16):
    # base_size =20
    plt.xlabel(xl, fontsize=base_size + 4)
    plt.ylabel(yl, fontsize=base_size + 4)
    plt.title(title, fontsize=base_size + 3)
    plt.legend(fontsize=base_size - 2, frameon=False)
    plt.xticks(fontsize=xtickx_size + 4)
    plt.tick_params(axis='both', which='major', labelsize=xtickx_size)
    plt.show()
# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# The following line improves formatting when ouputting NumPy arrays.
np.set_printoptions(linewidth=200)


# Define metrics component for tenser-flow to monitor the individual percentage
def percentage_error(output_index):
    def percentage_error(y_true, y_pred):
        # Selecting the specific output parameter based on output_index
        y_true_selected = y_true[:, output_index]
        y_pred_selected = y_pred[:, output_index]
        # Calculating percentage error
        error = tf.abs(
            (y_true_selected - y_pred_selected) / tf.clip_by_value(tf.abs(y_true_selected), 1e-6, tf.float32.max))
        return tf.reduce_mean(error) * 100

    percentage_error.__name__ = f'percentage_error_{output_index}'  # Naming the function

    return percentage_error


# Define plotting functions
def plot_curve(epochs, hist, list_of_metrics):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)
    plt.legend()


# Training functions:
def train_model(model, train_features, train_labels, epochs,
                batch_size=None, validation_split=0.1):
    """Train the model by feeding it data."""
    history = model.fit(x=train_features, y=train_labels, batch_size=batch_size,
                        epochs=epochs, shuffle=True,
                        validation_split=validation_split, verbose=0)
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    return epochs, hist


# return the epochs, full hist
def train_more(history, model, train_features, train_labels, additional_epochs,
               batch_size=None, validation_split_=0.1):
    additional_history = train_model(model, train_features, train_labels, additional_epochs, batch_size,
                                     validation_split=validation_split_)
    full_hist = pd.concat([history[1], additional_history[1]], ignore_index=True)
    return [range(len(full_hist)), full_hist]


# training_func takes history and additional epochs as input and output the new history
def live_train(training_func_, history_, step_, total_, list_of_metrics_):
    plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')

    for i in range(0, total_, step_):
        history_ = training_func_(history_, step_)
        # Clear the current figure and replot all metrics
        plt.clf()
        ax = plt.gca()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        for m in list_of_metrics_:
            x = history_[1][m]
            ax.plot(history_[0], x, label=m)
        ax.legend()
        # plt.yscale('log')
        plt.tight_layout()
        plt.pause(0.1)
        # Clear the previous output in Jupyter Notebook
        clear_output(wait=True)
    plt.ioff()
    plt.show()
    return history_


print("Ready!")
