import cv2
import numpy as np
import h5py
from tqdm import tqdm

mean = np.array([16.68440278, 38.07405572, 27.43923875, 22.9263001, 47.33313112, 42.46188265], dtype=np.float32).reshape((1, 1, 6))
std_dev_1 = np.array([1.0 / 17.76991678, 1.0 / 18.96145942, 1.0 / 21.60122086, 1.0 / 15.70177085, 1.0 / 17.90887029, 1.0 / 23.60007328], dtype=np.float32).reshape((1, 1, 6))


def denormalize(image):
    return (image / std_dev_1) + mean


if __name__ == '__main__':
    # file_path = "aug_de_norm_dataset.h5"
    #
    # h5file = h5py.File(file_path, 'r')
    # input_images = h5file['input_images']
    # output_images = h5file['output_images']
    #
    # count = 0
    #
    # for image, pred in tqdm(zip(input_images, output_images)):
    #     img = denormalize(image)
    #     diff = img[:, :, 3:] - img[:, :, :3]
    #     diff = diff[:, :, 2]
    #     val = np.max(diff)
    #
    #     if pred[0] == (val > 30.5): #35.5
    #         count += 1
    #
    # print("Accuracy: {:.2f}%".format(100 * count / len(input_images)))

    #avg 89.6

    before = cv2.imread(r"testing_events\Lac des Deux Cantons\Lac des Deux Cantons_before.png")
    after = cv2.imread(r"testing_events\Lac des Deux Cantons\Lac des Deux Cantons_after.png")

    diff = after[:, :, 2].astype(int) - before[:, :, 2].astype(int)
    diff[diff < 0] = 0
    diff = diff.astype(np.uint8)
    #diff = cv2.dilate(diff, np.ones((31, 31)), iterations=1)
    mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)[1]
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    #mask = cv2.dilate(mask, np.ones((31, 31)))
    mask = cv2.resize(mask, (1024, 1024))

    cv2.imshow("mask", mask)
    cv2.waitKey(0)

    # cv2.imshow("after", cv2.resize(after, (1024, 1024)))
    #
    # diff = (after.astype(int) - before.astype(int))
    # diff -= 25
    # diff[diff < 0] = 0
    # diff = diff[:, :, 2].astype(np.uint8)
    # #diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # diff = 255 * (diff > 0).astype(np.uint8)
    # diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    #
    # cv2.imshow("diff", cv2.resize(diff, (1024, 1024)))
    #
    # cv2.waitKey(0)