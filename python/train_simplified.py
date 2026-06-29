import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import h5py
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
from keras.callbacks import Callback
import numpy as np

import tensorflow as tf
from tensorflow import keras


class CheckpointsCallback(Callback):
    def __init__(self, checkpoints_path):
        super().__init__()
        self.checkpoints_path = checkpoints_path

    def on_epoch_end(self, epoch, logs=None):
        if self.checkpoints_path is not None:
            self.model.save_weights(self.checkpoints_path + "weights" + str(epoch+1) + ".h5")
            print("saved ", self.checkpoints_path + "weights" + str(epoch+1) + ".h5")


def load_h5_dataset(file_path, start, end, batch_size=16):
    # Function to load data from the HDF5 file
    def load_data(d_start, d_end):
        with h5py.File(file_path, 'r') as h5file:
            input_images = h5file['input_images'][d_start:d_end]
            output_images = h5file['output_images'][d_start:d_end]
            return input_images, output_images

    # Define a generator function to yield batches of data
    def data_generator():
        for batch_start in range(start, end, batch_size):
            batch_end = min(batch_start + batch_size, end)
            yield load_data(batch_start, batch_end)

    # Create a tf.data.Dataset from the generator
    output_signature = (
        tf.TensorSpec(shape=(None, 32, 32, 6), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    )
    dataset = tf.data.Dataset.from_generator(data_generator, output_signature=output_signature)
    dataset = dataset.map(lambda x, y: (x, y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def h5_size(data_path):
    with h5py.File(data_path, 'r') as h5file:
        return h5file['input_images'].shape[0]


def train_model(model, data_path, save_path, batch_size=128, epochs=50, w_path=''):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00001),
        loss=keras.losses.binary_crossentropy,
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()],
    )

    csv_logger = keras.callbacks.CSVLogger(save_path + "model_history_log.csv", append=True)

    callbacks = [
        CheckpointsCallback(save_path),
        csv_logger
    ]

    if w_path != '':
        model.load_weights(w_path)

    num_samples = h5_size(data_path)
    print("num_samples", num_samples)
    split_idx = int(num_samples * 0.8)

    train = load_h5_dataset(data_path, 0, split_idx,  batch_size)
    val = load_h5_dataset(data_path, split_idx, num_samples,  batch_size)

    return model.fit(train, validation_data=val, epochs=epochs, callbacks=callbacks)


def get_model():
    model = keras.models.Sequential()

    model.add(keras.applications.VGG19(input_shape=(32, 32, 6), weights=None, include_top=False))
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.summary()

    return model


if __name__ == '__main__':
    data_path = "aug_de_norm_dataset.h5"
    save_path = "experiments/vgg_19/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = get_model()

    train_model(model, data_path, save_path)

