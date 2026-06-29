import time

import os

import onnx
import tf2onnx

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import h5py
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
from keras.callbacks import Callback
import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import onnxruntime as ort
from CNN_models import ResNet18, VGG9, VGG_mini, VGG_micro, ShuffleNetV2, tiny_test
from fcn_models import make_fcn_vgg_binary
from transformer import swin_transformer

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


mean = np.array([16.68440278, 38.07405572, 27.43923875, 22.9263001, 47.33313112, 42.46188265], dtype=np.float32).reshape((1, 1, 6))
std_dev_1 = np.array([1.0 / 17.76991678, 1.0 / 18.96145942, 1.0 / 21.60122086, 1.0 / 15.70177085, 1.0 / 17.90887029, 1.0 / 23.60007328], dtype=np.float32).reshape((1, 1, 6))


def normalize(image):
    return (image - mean) * std_dev_1


def TverskyLoss(targets, inputs):
    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    # True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1.0 - targets) * inputs))
    FN = K.sum((targets * (1.0 - inputs)))

    Tversky = (TP + 1e-6) / (TP + 0.2 * FP + 0.8 * FN + 1e-6)

    return 1.0 - Tversky


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
    #print(model.layers)

    train_model(model, data_path, save_path)

    # model.load_weights(save_path + "weights49.h5") #27
    #
    # input_signature = [tf.TensorSpec(model.input_shape, tf.float32, name='tornado_patch_predictor_input')]
    #
    # print(model.input_shape)
    # print(input_signature)
    #
    # # Use from_function for tf functions
    # onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=18)
    # onnx.save(onnx_model, "tornado_patch_predictor_vgg_micro.onnx")


    # fp32_model = "tornado_patch_predictor_de_norm.onnx"
    # fp16_model = "tornado_patch_predictor_fp16.onnx"
    # #
    # # from onnxruntime.quantization import quantize_dynamic, QuantType
    # #
    # # quantize_dynamic(fp32_model, int8_model, weight_type=QuantType.QUInt8, per_channel=True)
    #
    # # from onnxconverter_common import float16
    # # import onnx
    # #
    # # model = onnx.load(fp32_model)
    # # model_fp16 = float16.convert_float_to_float16(model)
    # # onnx.save(model_fp16, fp16_model)

    # sess_options = ort.SessionOptions()
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # model_onnx = ort.InferenceSession("tornado_patch_predictor_vgg_micro.onnx", sess_options, providers=['CUDAExecutionProvider'])
    # input_name = model_onnx.get_inputs()[0].name
    # output_name = model_onnx.get_outputs()[0].name
    #
    # print(model_onnx.get_providers())
    #
    # before = cv2.imread(r"testing_events\Lac des Deux Cantons\Lac des Deux Cantons_before.png")
    # after = cv2.imread(r"testing_events\Lac des Deux Cantons\Lac des Deux Cantons_after.png")
    #
    # img = normalize(np.dstack((before, after)).astype(np.float32))
    #
    # count_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint16)
    # mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)
    #
    # patches = []
    # predictions = []
    # batch_size = 8192
    # stride = 32
    #
    # for i in tqdm(range(0, img.shape[0]-(32-stride), stride)):
    #     for j in range(0, img.shape[1]-(32-stride), stride):
    #         patches.append(img[i:i+32, j:j+32])
    #
    # predictions = []
    #
    # for i in tqdm(range(0, len(patches), batch_size)):
    #     predictions.extend([x[0] for x in model_onnx.run(None, {'tornado_patch_predictor_input': np.asarray(patches[i:min(i+batch_size, len(patches))], dtype=np.float32)})[0]])
    #
    # #predictions = model_onnx.run([output_name], {input_name: input_data})[0]
    #
    # idx = 0
    #
    # for i in tqdm(range(0, img.shape[0]-(32-stride), stride)):
    #     for j in range(0, img.shape[1]-(32-stride), stride):
    #
    #         mask[i:i+32, j:j+32] += predictions[idx]#[0]
    #         count_mask[i:i+32, j:j+32] += 1
    #         idx += 1
    #
    # mask /= count_mask
    # mask = 255 * (mask > 0.25).astype(np.uint8)
    # mask = cv2.dilate(mask, np.ones((31, 31)))
    # #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    #
    #
    # #ask = (2 * mask > count_mask).astype(np.uint8) * 255
    #
    # cv2.imwrite(save_path + "mask.png", mask)
    # mask = cv2.resize(mask, (1024, 1024))
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)






































