import os
from glob import glob
import cv2
from tqdm import tqdm
import numpy as np


def find_crop_location(large_image, small_image):
    result = cv2.matchTemplate(large_image, small_image, cv2.TM_CCOEFF_NORMED)
    return cv2.minMaxLoc(result)


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    # Load the image as a 1D array of bytes
    img_array = np.fromfile(path, dtype=np.uint8)
    # Decode the array into an image
    img = cv2.imdecode(img_array, flags)
    return img


def imwrite_unicode(path, img, extension=".png"):
    # Encode the image into a memory buffer
    result, encoded_img = cv2.imencode(extension, img)
    if result:
        # Write the buffer to the file
        with open(path, mode='wb') as f:
            encoded_img.tofile(f)
        return True
    return False


if __name__ == '__main__':

    input_folder = r"C:\Users\danie\Documents\Experiments\Satellite\saved"
    output_folder = r"C:\Users\danie\Documents\Experiments\Satellite\saved_patch_finder"
    events = os.listdir(input_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.makedirs(os.path.join(output_folder, "after"))
        os.makedirs(os.path.join(output_folder, "after_other"))
        os.makedirs(os.path.join(output_folder, "before"))
        os.makedirs(os.path.join(output_folder, "before_other"))
        count = 0
        other_count = 0
    else:
        count = len(os.listdir(os.path.join(output_folder, "after")))
        other_count = len(os.listdir(os.path.join(output_folder, "after_other")))

    print(count, other_count)

    for event in events[109:]:
        print(event)
        directory = os.path.join(input_folder, event)
        large_image_paths = sorted(glob(os.path.join(directory, "*.png")))

        patch_paths = glob(os.path.join(directory, "after", "*.png"))
        other_paths = glob(os.path.join(directory, "after_other", "*.png"))

        if large_image_paths is None or len(large_image_paths) != 2:
            continue

        after = imread_unicode(large_image_paths[0])
        before = imread_unicode(large_image_paths[1])

        large_patch_size = 256

        for patch_path in tqdm(patch_paths):
            patch = imread_unicode(patch_path)
            if patch is None:
                continue

            _, max_val, _, max_loc = find_crop_location(after, patch)

            if max_val < 0.99:
                continue

            patch_h, patch_w = patch.shape[:2]
            x, y = max_loc

            if (x < (large_patch_size // 2 - patch_w // 2) or y < (large_patch_size // 2 - patch_h // 2) or
                x + patch_w // 2 > after.shape[1] - large_patch_size // 2 or y + patch_h // 2 > after.shape[0] - large_patch_size // 2):
                continue

            after_patch = after[y - (large_patch_size // 2 - patch_h // 2):y + large_patch_size // 2 + patch_h // 2,
                                x - (large_patch_size // 2 - patch_w // 2):x + large_patch_size // 2 + patch_w // 2]
            before_patch = before[y - (large_patch_size // 2 - patch_h // 2):y + large_patch_size // 2 + patch_h // 2,
                                  x - (large_patch_size // 2 - patch_w // 2):x + large_patch_size // 2 + patch_w // 2]

            imwrite_unicode(os.path.join(output_folder, "after", f"{count}.png"), after_patch)
            imwrite_unicode(os.path.join(output_folder, "before", f"{count}.png"), before_patch)
            count += 1

        for patch_path in tqdm(other_paths):
            patch = imread_unicode(patch_path)
            if patch is None:
                continue

            _, max_val, _, max_loc = find_crop_location(after, patch)

            if max_val < 0.99:
                continue

            patch_h, patch_w = patch.shape[:2]
            x, y = max_loc

            if (x < (large_patch_size // 2 - patch_w // 2) or y < (large_patch_size // 2 - patch_h // 2) or
                x + patch_w // 2 > after.shape[1] - large_patch_size // 2 or y + patch_h // 2 > after.shape[0] - large_patch_size // 2):
                continue

            after_patch = after[y - (large_patch_size // 2 - patch_h // 2):y + large_patch_size // 2 + patch_h // 2,
                                x - (large_patch_size // 2 - patch_w // 2):x + large_patch_size // 2 + patch_w // 2]
            before_patch = before[y - (large_patch_size // 2 - patch_h // 2):y + large_patch_size // 2 + patch_h // 2,
                                  x - (large_patch_size // 2 - patch_w // 2):x + large_patch_size // 2 + patch_w // 2]

            imwrite_unicode(os.path.join(output_folder, "after_other", f"{other_count}.png"), after_patch)
            imwrite_unicode(os.path.join(output_folder, "before_other", f"{other_count}.png"), before_patch)
            other_count += 1



