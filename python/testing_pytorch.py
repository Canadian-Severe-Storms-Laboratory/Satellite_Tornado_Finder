from pathlib import Path

import cv2
import numpy as np
import onnx
from tqdm import tqdm
import torch

from train_pytorch import build_model


mean = np.array([20.95997633, 43.91433058, 35.6977732, 24.67067952, 49.58721762, 45.24592313], dtype=np.float32).reshape((1, 1, 6))
std_dev_1 = np.array([1.0 / 19.04020361, 1.0 / 21.74549397, 1.0 / 27.4036479, 1.0 / 19.91402317, 1.0 / 22.74246173, 1.0 / 29.72733925], dtype=np.float32).reshape((1, 1, 6))


def normalize(image):
    return (image - mean) * std_dev_1


def predict_batch(batch):
    batch = torch.from_numpy(batch).to('cuda')

    preds = model(batch).detach().cpu().numpy().flatten()

    return preds


if __name__ == '__main__':
    event_name = "Lac Pedro"
    before = cv2.imread(f"testing_events/{event_name}/{event_name}_before.png")
    after  = cv2.imread(f"testing_events/{event_name}/{event_name}_after.png")
    img    = normalize(np.dstack((before, after))).astype(np.float32)

    # patch_model = build_model()
    # patch_model.load_weights("experiments/aug_de_norm_test1/weights20.h5")

    model = build_model()
    device = torch.device('cuda')
    model.to(device)

    state = torch.load(Path("weights6.pt"), map_location='cuda')
    model.load_state_dict(state)
    model.eval()

    dummy_input = torch.randn(256, 6, 64, 64).to(device)

    dynamic_axes = {
        'input_name': {0: 'batch_size'},  # 'input_name' is a placeholder
        'output_name': {0: 'batch_size'}  # 'output_name' is a placeholder
    }

    onnx_program = torch.onnx.export(model, dummy_input, dynamo=True, input_names=['input_name'], output_names=['output_name'], dynamic_axes={'input_name': {0: 'batch_size'}, 'output_name': {0: 'batch_size'}})
    #torch.onnx.export(model, dummy_input, "model64.onnx", input_names=['tornado_patch_predictor_input'], output_names=['output'], dynamic_axes={'tornado_patch_predictor_input': {0: 'batch_size'}})
    onnx_program.save("model64.onnx")

    count_mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)

    patches = []
    predictions = []

    for i in tqdm(range(0, img.shape[0]-56, 8)):
        for j in range(0, img.shape[1]-56, 8):
            patches.append(np.transpose(img[i:i + 64, j:j + 64], (2, 0, 1)).astype(np.float32).reshape((6, 64, 64)))

    predictions = []

    for i in tqdm(range(0, len(patches), 1024)):
        #predictions.extend([x[0] for x in model_onnx.run(None, {'tornado_patch_predictor_input': np.asarray(patches[i:min(i + 1024, len(patches))], dtype=np.float32)})[0]])
        predictions.extend(predict_batch(np.asarray(patches[i:min(i + 1024, len(patches))], dtype=np.float32)))

    # predictions = model_onnx.run([output_name], {input_name: input_data})[0]

    idx = 0

    for i in tqdm(range(0, img.shape[0]-56, 8)):
        for j in range(0, img.shape[1]-56, 8):
            mask[i:i + 64, j:j + 64] += predictions[idx]  # [0]
            count_mask[i:i + 64, j:j + 64] += 1
            idx += 1

    mask /= count_mask
    mask = 255 * (mask > 0.7).astype(np.uint8)

    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)

    pmask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("mask", pmask)
    cv2.waitKey(0)