import csv
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as tvm
from torcheval.metrics import BinaryAccuracy, BinaryPrecision, BinaryRecall
import sys
from tqdm import tqdm

from utils import CheckpointSaver, h5_size, H5BinaryClassificationDataset
from time import perf_counter


def train_model(
    model:          nn.Module,
    data_path:      str,
    save_path:      str,
    batch_size:     int = 32,
    epochs:         int = 50,
    weights_init:   str = '',          # optional .pt file to warm-start
    device:         str | torch.device = 'cuda' if torch.cuda.is_available() else 'cpu'
):

    device   = torch.device(device)
    model    = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-6)
    ckpt_saver = CheckpointSaver(save_path)

    os.makedirs(save_path, exist_ok=True)
    log_file = os.path.join(save_path, "model_history_log.csv")
    csv_header = ["epoch", "loss", "accuracy", "precision", "recall", "val_loss", "val_accuracy", "val_precision", "val_recall"]
    first_write = not os.path.exists(log_file)

    logfile = open(log_file, 'a', newline='')
    csv_logger = csv.writer(logfile)
    if first_write:
        csv_logger.writerow(csv_header)

    if weights_init:
        model.load_state_dict(torch.load(weights_init, map_location=device))
        print(f"Loaded weights from {weights_init}")

    n_samples = h5_size(data_path)
    split_idx = int(0.8 * n_samples)

    train_ds = H5BinaryClassificationDataset(data_path, 0,           split_idx)
    val_ds   = H5BinaryClassificationDataset(data_path, split_idx,   n_samples)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    # pos_weight = torch.tensor([2.0], device=device)
    # loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_func = nn.BCELoss()
    metrics = [BinaryAccuracy(), BinaryPrecision(), BinaryRecall()]
    val_metrics = [BinaryAccuracy(), BinaryPrecision(), BinaryRecall()]

    for epoch in range(epochs):
        # -------- training --------------------------------------------------
        model.train()
        epoch_loss = 0.0

        for m in metrics:
            m.reset()

        for xb, yb in tqdm(train_loader, file=sys.stdout, leave=False):
            xb, yb = xb.to(device), yb.to(device)

            optimiser.zero_grad(set_to_none=True)
            preds = model(xb)

            loss = loss_func(preds, yb)
            loss.backward()
            optimiser.step()

            n = xb.size(0)
            epoch_loss += loss.item() * n

            preds = preds.flatten() > 0.5
            yb = yb.flatten() > 0.5

            for i in range(len(metrics)):
                metrics[i].update(preds, yb)

        epoch_loss /= len(train_ds)

        # -------- validation ------------------------------------------------
        model.eval()
        val_loss = 0.0

        for m in val_metrics:
            m.reset()

        with torch.no_grad():
            for xb, yb in tqdm(val_loader, file=sys.stdout, leave=False):
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss  = loss_func(preds, yb)

                n = xb.size(0)
                val_loss += loss.item() * n

                preds = preds.flatten() > 0.5
                yb = yb.flatten() > 0.5

                for i in range(len(metrics)):
                    val_metrics[i].update(preds, yb)

        val_loss /= len(val_ds)

        # -------- logging / checkpoint -------------------------------------
        print(f"[{epoch+1:03}/{epochs}] loss {epoch_loss:.4f}", end=" ")

        for i in range(len(metrics)):
            print(f"| {metrics[i].__class__.__name__} {metrics[i].compute():.4f}", end=" ")

        print(f"| val_loss {val_loss:.4f}", end=" ")

        for i in range(len(metrics)):
            print(f"| val_{metrics[i].__class__.__name__} {val_metrics[i].compute():.4f}", end=" ")

        csv_logger.writerow(
            [epoch+1, epoch_loss, *[m.compute() for m in metrics], val_loss, *[m.compute() for m in val_metrics]]
        )
        logfile.flush()

        ckpt_saver(model, epoch)

    logfile.close()
    print("✓ training finished")


def build_model():
    model = tvm.vgg19_bn(weights=None)

    stem_conv = model.features[0]

    new_conv = nn.Conv2d(
        12,
        stem_conv.out_channels,
        kernel_size=stem_conv.kernel_size,
        stride=stem_conv.stride,
        padding=stem_conv.padding,
        bias=(stem_conv.bias is not None),
    )

    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

    if new_conv.bias is not None:
        nn.init.zeros_(new_conv.bias)

    model.features[0] = new_conv

    in_feats = model.classifier[0].in_features

    model.classifier = nn.Sequential(
        nn.Linear(in_feats, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )

    return model


if __name__ == "__main__":
    model = build_model()

    data_path = "dataset64_256_4.h5"
    save_path = "experiments/vgg19_64_256/"
    weights_path = "experiments/test64_256_4/weights29.pt"

    train_model(model, data_path, save_path, batch_size=128, epochs=100, weights_init=weights_path)

