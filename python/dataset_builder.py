import os
import random
from glob import glob
import numpy as np
import h5py
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from scipy.stats import norm


def read_images_in_directory(directory):
    images = []

    files = glob(os.path.join(directory, '*.png'))

    for file in files:
        images.append(cv2.imread(file))

    return images


def count_images_in_directory(directory):
    files = glob(os.path.join(directory, '*.png'))
    return len(files)


def random_brightness_contrast(img):
    alpha = np.random.uniform(0.8, 1.2)  # Contrast control
    beta = np.random.randint(-20, 20)  # Brightness control

    # Apply brightness and contrast adjustment
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted


def random_hue_adjust(img, max_delta=20):

    # Convert BGR to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Extract hue channel
    h, s, v = cv2.split(hsv_img)

    # Generate a random hue shift
    hue_shift = random.randint(-10, 30)

    # Apply the hue shift, ensuring it wraps around 0-179
    h_shifted = (h.astype(int) + hue_shift) % 180

    # Merge channels back and convert to BGR
    adjusted_hsv_img = cv2.merge([h_shifted.astype(np.uint8), s, v])
    return cv2.cvtColor(adjusted_hsv_img, cv2.COLOR_HSV2BGR)


def aug_image(image, type):
    transformations = [
        image,  # Original
        np.fliplr(image),  # Horizontal flip
        np.flipud(image),  # Vertical flip
        np.transpose(image, axes=(1, 0, 2)),  # Diagonal flip (main diagonal)
        np.fliplr(np.transpose(image, axes=(1, 0, 2))),  # Diagonal flip (anti-diagonal)
        np.rot90(image, k=1),  # 90 degrees
        np.rot90(image, k=2),  # 180 degrees
        np.rot90(image, k=3)  # 270 degrees
    ]

    if type == 0:
        transformations = random.sample(transformations, 3)

    return np.asarray(transformations)


#mean = np.array([16.68440278, 38.07405572, 27.43923875, 22.9263001, 47.33313112, 42.46188265], dtype=np.float32).reshape((1, 1, 6))
#std_dev_1 = np.array([1.0 / 17.76991678, 1.0 / 18.96145942, 1.0 / 21.60122086, 1.0 / 15.70177085, 1.0 / 17.90887029, 1.0 / 23.60007328], dtype=np.float32).reshape((1, 1, 6))

mean = np.array([20.95997633, 43.91433058, 35.6977732, 24.67067952, 49.58721762, 45.24592313], dtype=np.float32).reshape((1, 1, 6))
std_dev_1 = np.array([1.0 / 19.04020361, 1.0 / 21.74549397, 1.0 / 27.4036479, 1.0 / 19.91402317, 1.0 / 22.74246173, 1.0 / 29.72733925], dtype=np.float32).reshape((1, 1, 6))


def normalize(image):
    return (image - mean) * std_dev_1


def compute_mean_std(images):
    # Concatenate all images into one large array of shape (total_pixels, 3)
    all_pixels = np.concatenate([img.reshape(-1, 3) for img in images], axis=0)

    # Compute mean and std across the pixel dimension (axis=0 -> B, G, R)
    mean_vals = np.mean(all_pixels, axis=0)
    std_vals = np.std(all_pixels, axis=0)

    return mean_vals, std_vals


if __name__ == '__main__':
    path = "C:/Users/danie/Documents/Experiments/Satellite/saved"

    events = [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    names = [os.path.basename(event) for event in events]

    before_imgs = []
    after_imgs = []
    other_before_imgs = []
    other_after_imgs = []

    # count = 0
    # other_count = 0
    #
    # for event in tqdm(events):
    #     count += count_images_in_directory(os.path.join(event, "before"))
    #     other_count += count_images_in_directory(os.path.join(event, "before_other"))
    #
    # print(count, other_count)

    event_counts = []

    for event in tqdm(events):
        event_counts.append([os.path.basename(event), count_images_in_directory(os.path.join(event, "before"))])

    sorted_event_counts = sorted(event_counts, key=lambda x: x[1], reverse=True)

    for event, count in sorted_event_counts:
        print(f"{event}: {count} images")

    for event in tqdm(events):

        before_imgs.extend(read_images_in_directory(os.path.join(event, "before")))
        after_imgs.extend(read_images_in_directory(os.path.join(event, "after")))
        other_before_imgs.extend(read_images_in_directory(os.path.join(event, "before_other")))
        other_after_imgs.extend(read_images_in_directory(os.path.join(event, "after_other")))

    #other_before_imgs.extend(read_images_in_directory(os.path.join("additional_other", "before_other")))
    #other_after_imgs.extend(read_images_in_directory(os.path.join("additional_other", "after_other")))

    #print(compute_mean_std(before_imgs + other_before_imgs))
    #print(compute_mean_std(after_imgs + other_after_imgs))

    h5_file_path = 'dataset64.h5'
    dataset = []
    data_length = 0


    for i in tqdm(range(len(before_imgs))):
        input_img = normalize(np.dstack((before_imgs[i], after_imgs[i])).astype('float32'))

        dataset.append([input_img, 1.0])
        data_length += 8

    for i in tqdm(range(len(other_before_imgs))):
        input_img = normalize(np.dstack((other_before_imgs[i], other_after_imgs[i])).astype('float32'))

        dataset.append([input_img, 0.0])
        data_length += 3

    random.shuffle(dataset)

    num_samples = len(dataset)

    with h5py.File(h5_file_path, 'w') as h5file:

        input_shape = (64, 64, 6)
        output_shape = (1,)

        h5file.create_dataset('input_images', shape=(data_length,) + input_shape, dtype='float32')
        h5file.create_dataset('output_images', shape=(data_length,) + output_shape, dtype='float32')

        idx = 0

        for i in tqdm(range(num_samples)):
            aug = aug_image(dataset[i][0], dataset[i][1])

            if dataset[i][1] == 1:
                h5file['input_images'][idx:idx+8] = aug
                h5file['output_images'][idx:idx+8] = dataset[i][1]

                idx += 8

            else:
                h5file['input_images'][idx:idx+3] = aug
                h5file['output_images'][idx:idx+3] = dataset[i][1]

                idx += 3


















