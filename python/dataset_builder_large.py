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


def random_brightness_contrast(imgs):
    alpha = np.random.uniform(0.9, 1.1)  # Contrast control
    beta = np.random.randint(-10, 10)  # Brightness control

    adjusted_imgs = []
    for img in imgs:
        # Apply brightness and contrast adjustment
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        adjusted_imgs.append(adjusted)
    return adjusted_imgs


def random_hue_adjust(imgs, max_delta=20):
    # Generate a random hue shift once for the entire list
    hue_shift = random.randint(-10, 30)

    adjusted_imgs = []
    for img in imgs:
        # Convert BGR to HSV
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Extract hue channel
        h, s, v = cv2.split(hsv_img)

        # Apply the hue shift, ensuring it wraps around 0-179
        h_shifted = (h.astype(int) + hue_shift) % 180

        # Merge channels back and convert to BGR
        adjusted_hsv_img = cv2.merge([h_shifted.astype(np.uint8), s, v])
        adjusted_imgs.append(cv2.cvtColor(adjusted_hsv_img, cv2.COLOR_HSV2BGR))

    return adjusted_imgs


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
        transformations = random.sample(transformations, 4)

    return np.asarray(transformations)


#mean = np.array([16.68440278, 38.07405572, 27.43923875, 22.9263001, 47.33313112, 42.46188265], dtype=np.float32).reshape((1, 1, 6))
#std_dev_1 = np.array([1.0 / 17.76991678, 1.0 / 18.96145942, 1.0 / 21.60122086, 1.0 / 15.70177085, 1.0 / 17.90887029, 1.0 / 23.60007328], dtype=np.float32).reshape((1, 1, 6))

# mean = np.array([20.95997633, 43.91433058, 35.6977732, 24.67067952, 49.58721762, 45.24592313], dtype=np.float32).reshape((1, 1, 6))
# std_dev_1 = np.array([1.0 / 19.04020361, 1.0 / 21.74549397, 1.0 / 27.4036479, 1.0 / 19.91402317, 1.0 / 22.74246173, 1.0 / 29.72733925], dtype=np.float32).reshape((1, 1, 6))

mean = np.array([18.68838881, 41.17134002, 31.97962192, 19.05978276, 43.07571633, 35.68422487, 18.68838881, 41.17134002, 31.97962192, 19.05978276, 43.07571633, 35.68422487], dtype=np.float32).reshape((1, 1, 12))
std_dev_1 = np.array([1.0 / 15.62747299, 1.0 / 18.76652384, 1.0 / 23.22864492, 1.0 / 15.70466111, 1.0 / 19.25905989, 1.0 / 24.46462193, 1.0 / 15.62747299, 1.0 / 18.76652384, 1.0 / 23.22864492, 1.0 / 15.70466111, 1.0 / 19.25905989, 1.0 / 24.46462193], dtype=np.float32).reshape((1, 1, 12))


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


    # print(compute_mean_std(before_imgs + other_before_imgs))
    # print(compute_mean_std(after_imgs + other_after_imgs))

    h5_file_path = 'dataset64_256_4.h5'
    dataset = []
    data_length = 0

    patch_size = 64
    large_patch_size = 256
    count_reg = 0
    count_other = 0

    print("Loading EU dataset...")
    path = "C:/Users/danie/Documents/Experiments/Satellite/Saved_EU"
    for folder in tqdm(os.listdir(path)):
        before_imgs = read_images_in_directory(os.path.join(path, folder, "before_large"))
        after_imgs = read_images_in_directory(os.path.join(path, folder, "after_large"))
        other_before_imgs = read_images_in_directory(os.path.join(path, folder, "before_other_large"))
        other_after_imgs = read_images_in_directory(os.path.join(path, folder, "after_other_large"))

        for i in range(len(before_imgs)):
            before, after = random_brightness_contrast([before_imgs[i], after_imgs[i]])

            if before is None or after is None:
                continue

            before_small = before[
                large_patch_size // 2 - patch_size // 2:large_patch_size // 2 + patch_size // 2, large_patch_size // 2 - patch_size // 2:large_patch_size // 2 + patch_size // 2]
            after_small = after[
                large_patch_size // 2 - patch_size // 2:large_patch_size // 2 + patch_size // 2, large_patch_size // 2 - patch_size // 2:large_patch_size // 2 + patch_size // 2]

            before_large = cv2.resize(before, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
            after_large = cv2.resize(after, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

            input_img = normalize(np.dstack((before_small, after_small, before_large, after_large)).astype('float32'))

            dataset.append([input_img, 1.0])
            count_reg += 1
            data_length += 8

        for i in range(len(other_before_imgs)):
            before, after = random_brightness_contrast([other_before_imgs[i], other_after_imgs[i]])

            if before is None or after is None:
                continue

            before_small = before[
                large_patch_size // 2 - patch_size // 2:large_patch_size // 2 + patch_size // 2, large_patch_size // 2 - patch_size // 2:large_patch_size // 2 + patch_size // 2]
            after_small = after[
                large_patch_size // 2 - patch_size // 2:large_patch_size // 2 + patch_size // 2, large_patch_size // 2 - patch_size // 2:large_patch_size // 2 + patch_size // 2]

            before_large = cv2.resize(before, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
            after_large = cv2.resize(after, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

            input_img = normalize(np.dstack((before_small, after_small, before_large, after_large)).astype('float32'))

            dataset.append([input_img, 0.0])
            count_other += 1
            data_length += 4

    #load downburst dataset
    print("Loading downburst dataset...")
    path = "C:/Users/danie/Documents/Experiments/Satellite/savedV3"
    for folder in tqdm(os.listdir(path)):
        before_imgs = read_images_in_directory(os.path.join(path, folder, "before_large"))
        after_imgs = read_images_in_directory(os.path.join(path, folder, "after_large"))
        other_before_imgs = read_images_in_directory(os.path.join(path, folder, "before_other_large"))
        other_after_imgs = read_images_in_directory(os.path.join(path, folder, "after_other_large"))

        for i in range(len(before_imgs)):
            before, after = random_brightness_contrast([before_imgs[i], after_imgs[i]])

            if before is None or after is None:
                continue

            before_small = before[large_patch_size // 2 - patch_size // 2:large_patch_size // 2 + patch_size // 2, large_patch_size // 2 - patch_size // 2:large_patch_size // 2 + patch_size // 2]
            after_small = after[large_patch_size // 2 - patch_size // 2:large_patch_size // 2 + patch_size // 2, large_patch_size // 2 - patch_size // 2:large_patch_size // 2 + patch_size // 2]

            before_large = cv2.resize(before, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
            after_large = cv2.resize(after, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

            input_img = normalize(np.dstack((before_small, after_small, before_large, after_large)).astype('float32'))

            dataset.append([input_img, 1.0])
            count_reg += 1
            data_length += 8

        for i in range(len(other_before_imgs)):
            before, after = random_brightness_contrast([other_before_imgs[i], other_after_imgs[i]])

            if before is None or after is None:
                continue

            before_small = before[large_patch_size // 2 - patch_size // 2:large_patch_size // 2 + patch_size // 2, large_patch_size // 2 - patch_size // 2:large_patch_size // 2 + patch_size // 2]
            after_small = after[large_patch_size // 2 - patch_size // 2:large_patch_size // 2 + patch_size // 2, large_patch_size // 2 - patch_size // 2:large_patch_size // 2 + patch_size // 2]

            before_large = cv2.resize(before, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
            after_large = cv2.resize(after, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

            input_img = normalize(np.dstack((before_small, after_small, before_large, after_large)).astype('float32'))

            dataset.append([input_img, 0.0])
            count_other += 1
            data_length += 4

    # load tornado dataset
    print("Loading tornado dataset...")
    path = "C:/Users/danie/Documents/Experiments/Satellite/saved_patch_finder"
    before_imgs = read_images_in_directory(os.path.join(path, "before"))
    after_imgs = read_images_in_directory(os.path.join(path, "after"))
    other_before_imgs = read_images_in_directory(os.path.join(path, "before_other"))
    other_after_imgs = read_images_in_directory(os.path.join(path, "after_other"))

    for i in tqdm(range(len(before_imgs))):
        before, after = random_brightness_contrast([before_imgs[i], after_imgs[i]])

        before_small = before[
            large_patch_size // 2 - patch_size // 2:large_patch_size // 2 + patch_size // 2, large_patch_size // 2 - patch_size // 2:large_patch_size // 2 + patch_size // 2]
        after_small = after[
            large_patch_size // 2 - patch_size // 2:large_patch_size // 2 + patch_size // 2, large_patch_size // 2 - patch_size // 2:large_patch_size // 2 + patch_size // 2]

        before_large = cv2.resize(before, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
        after_large = cv2.resize(after, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

        input_img = normalize(np.dstack((before_small, after_small, before_large, after_large)).astype('float32'))

        dataset.append([input_img, 1.0])
        count_reg += 1
        data_length += 8

    for i in tqdm(range(len(other_before_imgs))):
        before, after = random_brightness_contrast([other_before_imgs[i], other_after_imgs[i]])

        before_small = before[
            large_patch_size // 2 - patch_size // 2:large_patch_size // 2 + patch_size // 2, large_patch_size // 2 - patch_size // 2:large_patch_size // 2 + patch_size // 2]
        after_small = after[
            large_patch_size // 2 - patch_size // 2:large_patch_size // 2 + patch_size // 2, large_patch_size // 2 - patch_size // 2:large_patch_size // 2 + patch_size // 2]

        before_large = cv2.resize(before, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
        after_large = cv2.resize(after, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

        input_img = normalize(np.dstack((before_small, after_small, before_large, after_large)).astype('float32'))

        dataset.append([input_img, 0.0])
        count_other += 1
        data_length += 4

    print(f"Forest: {count_reg}, Other: {count_other}")

    random.shuffle(dataset)

    num_samples = len(dataset)

    with h5py.File(h5_file_path, 'w') as h5file:

        input_shape = (64, 64, 12)
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
                h5file['input_images'][idx:idx+4] = aug
                h5file['output_images'][idx:idx+4] = dataset[i][1]

                idx += 4


















