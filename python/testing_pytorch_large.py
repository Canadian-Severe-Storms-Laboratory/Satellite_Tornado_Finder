from pathlib import Path

import cv2
import numpy as np
import onnx
from tqdm import tqdm
import torch

from train_pytorch import build_model


mean = np.array([18.68838881, 41.17134002, 31.97962192, 19.05978276, 43.07571633, 35.68422487], dtype=np.float32).reshape((1, 1, 6))
std_dev_1 = np.array([1.0 / 15.62747299, 1.0 / 18.76652384, 1.0 / 23.22864492, 1.0 / 15.70466111, 1.0 / 19.25905989, 1.0 / 24.46462193], dtype=np.float32).reshape((1, 1, 6))


def normalize(image):
    return (image - mean) * std_dev_1


def predict_batch(batch):
    batch = torch.from_numpy(batch).to('cuda')

    preds = model(batch).detach().cpu().numpy().flatten()

    return preds


if __name__ == '__main__':
    event_name = "Lac Flocon"
    before = cv2.imread(f"testing_events/{event_name}/{event_name}_before.png")
    after  = cv2.imread(f"testing_events/{event_name}/{event_name}_after.png")
    img    = normalize(np.dstack((before, after))).astype(np.float32)

    # patch_model = build_model()
    # patch_model.load_weights("experiments/aug_de_norm_test1/weights20.h5")

    model = build_model()
    device = torch.device('cuda')
    model.to(device)

    state = torch.load(Path("weights85.pt"), map_location='cuda')
    model.load_state_dict(state)
    model.eval()

    # dummy_input = torch.randn(256, 12, 64, 64).to(device)
    #
    # dynamic_axes = {
    #     'input_name': {0: 'batch_size'},  # 'input_name' is a placeholder
    #     'output_name': {0: 'batch_size'}  # 'output_name' is a placeholder
    # }
    #
    # onnx_program = torch.onnx.export(model, dummy_input, dynamo=True, input_names=['input_name'], output_names=['output_name'], dynamic_axes={'input_name': {0: 'batch_size'}, 'output_name': {0: 'batch_size'}})
    # #torch.onnx.export(model, dummy_input, "model64.onnx", input_names=['tornado_patch_predictor_input'], output_names=['output'], dynamic_axes={'tornado_patch_predictor_input': {0: 'batch_size'}})
    # onnx_program.save("model64_256.onnx")

    count_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)

    patches = []
    predictions = []

    patch_size = 64
    larger_patch_size = 256
    stride = 16

    for i in tqdm(range(larger_patch_size // 2 - patch_size // 2, img.shape[0]-(larger_patch_size // 2 + patch_size // 2), stride)):
        for j in range(larger_patch_size // 2 - patch_size // 2, img.shape[0]-(larger_patch_size // 2 + patch_size // 2), stride):

            before_large = img[i:i + larger_patch_size, j:j + larger_patch_size, 0:3]
            after_large  = img[i:i + larger_patch_size, j:j + larger_patch_size, 3:6]
            before = before_large[larger_patch_size // 2 - patch_size // 2:larger_patch_size // 2 + patch_size // 2, larger_patch_size // 2 - patch_size // 2:larger_patch_size // 2 + patch_size // 2]
            after  = after_large[larger_patch_size // 2 - patch_size // 2:larger_patch_size // 2 + patch_size // 2, larger_patch_size // 2 - patch_size // 2:larger_patch_size // 2 + patch_size // 2]
            before_large = cv2.resize(before_large, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
            after_large  = cv2.resize(after_large, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

            patch = np.dstack((before, after, before_large, after_large)).astype(np.float32)

            patches.append(np.transpose(patch, (2, 0, 1)).reshape((12, 64, 64)))

    predictions = []

    for i in tqdm(range(0, len(patches), 1024)):
        #predictions.extend([x[0] for x in model_onnx.run(None, {'tornado_patch_predictor_input': np.asarray(patches[i:min(i + 1024, len(patches))], dtype=np.float32)})[0]])
        predictions.extend(predict_batch(np.asarray(patches[i:min(i + 1024, len(patches))], dtype=np.float32)))

    # predictions = model_onnx.run([output_name], {input_name: input_data})[0]

    idx = 0

    for i in tqdm(range(larger_patch_size // 2 - patch_size // 2, img.shape[0] - (larger_patch_size // 2 + patch_size // 2), stride)):
        for j in range(larger_patch_size // 2 - patch_size // 2, img.shape[0] - (larger_patch_size // 2 + patch_size // 2), stride):
            mask[i:i + 64, j:j + 64] += predictions[idx]  # [0]
            count_mask[i:i + 64, j:j + 64] += 1
            idx += 1

    mask /= count_mask
    mask = (mask * 255).astype(np.uint8)
    # mask = 255 * (mask > 0.7).astype(np.uint8)
    #
    # mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)

    pmask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_AREA)

    cv2.imshow("mask", pmask)
    cv2.waitKey(0)