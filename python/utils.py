from random import randint

import cv2
import h5py
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class CheckpointSaver:
    def __init__(self, checkpoints_path: str | None):
        self.checkpoints_path = checkpoints_path
        if checkpoints_path:
            os.makedirs(checkpoints_path, exist_ok=True)

    def __call__(self, model: nn.Module, epoch: int) -> None:
        if not self.checkpoints_path:
            return
        fname = os.path.join(self.checkpoints_path, f"weights{epoch+1}.pt")
        torch.save(model.state_dict(), fname)
        print(f"saved {fname}")


class H5BinaryClassificationDataset(Dataset):

    def __init__(self, file_path: str, start: int, end: int):
        self.file_path = file_path
        self.indices = list(range(start, end))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        with h5py.File(self.file_path, 'r') as h5file:
            x = h5file['input_images'][i]
            y = h5file['output_images'][i]

        # TF/keras → channels-first for PyTorch
        x = np.transpose(x, (2, 0, 1)).astype(np.float32)
        y = y.astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)


def h5_size(data_path: str) -> int:
    with h5py.File(data_path, 'r') as h5file:
        return h5file['input_images'].shape[0]


def count_files_non_recursive(directory_path):
    file_count = 0
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isfile(item_path):
            file_count += 1
    return file_count


def cosine_distances(A, B):
    S = A @ B.T
    D = 1.0 - S
    return D


def rotate_image(image, angle, flags):
    height, width = image.shape[:2]
    center = (width / 2, height / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0) # scale=1.0 for original size

    rotated_image = cv2.warpAffine(image, M, (width, height), flags=flags)

    return rotated_image


def translate_and_rotate(img, point, angle_deg, scale=1.0, border_mode=cv2.BORDER_CONSTANT, border_value=0):
    h, w = img.shape[:2]
    cx, cy = w * 0.5, h * 0.5

    # translation matrix, point -> centre
    dx, dy = cx - point[0], cy - point[1]
    T = np.array([[1, 0, dx],
                  [0, 1, dy],
                  [0, 0, 1]], dtype=np.float32)

    # rotation matrix about the unchanged centre
    R = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
    R = np.vstack([R, [0, 0, 1]]).astype(np.float32)

    # first translate, then rotate
    M = R @ T
    M = M[:2]

    transformed = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=border_mode, borderValue=border_value)

    return transformed


def aug_image(image):
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

    return transformations


def random_hue_adjust(img, max_delta=20):

    # Convert BGR to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Extract hue channel
    h, s, v = cv2.split(hsv_img)

    # Generate a random hue shift
    hue_shift = randint(-10, 30)

    # Apply the hue shift, ensuring it wraps around 0-179
    h_shifted = (h.astype(int) + hue_shift) % 180

    # Merge channels back and convert to BGR
    adjusted_hsv_img = cv2.merge([h_shifted.astype(np.uint8), s, v])
    return cv2.cvtColor(adjusted_hsv_img, cv2.COLOR_HSV2BGR)


def min_rotation_to_e1(v):
    v = v.astype(np.float64)
    c = v[0]
    N = v.size
    e1 = np.zeros_like(v)
    e1[0] = 1.0

    L = np.outer(e1, v) - np.outer(v, e1)
    R = np.eye(N) + L + (L @ L) / (1.0 + c)
    return R


def rotate_embeddings(embeddings, R):
    flat = embeddings.reshape(-1, 16).T
    rotated_flat = R @ flat
    return rotated_flat.T.reshape(embeddings.shape).astype(np.float32)


def min_pool_first(img: np.ndarray, pool_size: int = 2) -> np.ndarray:
    p = pool_size
    H, W, D = img.shape
    Hc, Wc = (H // p) * p, (W // p) * p
    hb, wb = Hc // p, Wc // p

    tiles = img.reshape(hb, p, wb, p, D)

    tiles = tiles.swapaxes(1, 2)

    tiles_flat = tiles.reshape(hb, wb, p * p, D)
    first_flat = tiles_flat[..., 0]

    idx = np.argmin(first_flat, axis=-1)[..., None, None]

    pooled = np.take_along_axis(tiles_flat, idx, axis=2).squeeze(axis=2)

    return pooled

