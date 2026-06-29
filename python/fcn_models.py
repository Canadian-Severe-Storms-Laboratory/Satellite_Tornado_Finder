import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from time import perf_counter


def make_fcn_vgg_binary(
        input_shape=(32, 32, 6),
        stride: int = 8,
        use_imagenet_weights: bool = True,
        output_mode = "training"
) -> tf.keras.Model:

    if stride not in (4, 8, 16, 32):
        raise ValueError("stride must be one of {4, 8, 16, 32}")

    def conv_bn_relu(x, filters, k=3):
        x = layers.Conv2D(filters, k, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        return layers.Activation("relu")(x)

    def vgg_block(x, filters, n_conv, pool_stride):
        for _ in range(n_conv):
            x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)#conv_bn_relu(x, filters)
        x = layers.MaxPooling2D(2, strides=pool_stride, padding="same")(x)
        return x

    def duplicate_kernel_rgb_to_six(kernel_rgb: np.ndarray) -> np.ndarray:
        k_h, k_w, _, n_out = kernel_rgb.shape
        kernel_6 = np.concatenate([kernel_rgb, kernel_rgb], axis=2) * 0.5
        assert kernel_6.shape == (k_h, k_w, 6, n_out)
        return kernel_6

    # ── build the network ────────────────────────────────────────────────
    inputs = layers.Input(shape=input_shape)
    x = inputs

    pool_plan = [2, 2, 2, 2, 2]      # default stride 32
    if stride <= 16:
        pool_plan[4] = 1
    if stride <= 8:
        pool_plan[3] = 1
    if stride <= 4:
        pool_plan[2] = 1

    x = vgg_block(x, 64, 2, pool_plan[0])
    x = vgg_block(x, 128, 2, pool_plan[1])
    x = vgg_block(x, 256, 3, pool_plan[2])
    x = vgg_block(x, 512, 3, pool_plan[3])
    x = vgg_block(x, 512, 3, pool_plan[4])

    # classifier: 1×1 convs so it also works on tiny inputs
    #x = layers.Conv2D(4096, 1, activation="relu")(x)
    x = layers.Conv2D(256, 1, activation="relu")(x)
    #x = conv_bn_relu(x, 256, k=1)
    #x = conv_bn_relu(x, 4096, k=1)
    logits = layers.Conv2D(1, 1, name="logits")(x)

    if output_mode == "pred":
        out = layers.Activation("sigmoid", name="prob")(logits)  # (H/s, W/s, 1)
    else:
        pooled = layers.GlobalAveragePooling2D()(logits)  # (1,)
        out = layers.Activation("sigmoid", name="prob")(pooled)  # (1,)

    model = models.Model(inputs, out, name=f"FCNVGG16{stride}")

    # ── weight transplant (optional) ──────────────────────────────────────
    if use_imagenet_weights:
        from tensorflow.keras.applications import VGG16
        src = VGG16(include_top=False, weights='imagenet')
        #src.summary()

        # copy everything *except* the first conv layer (channel mismatch)
        name2dst = {l.name: l for l in model.layers}
        for layer in src.layers:
            if layer.name == 'block1_conv1':
                # adapt RGB→6‑channel weights
                W_rgb, b_rgb = layer.get_weights()      # (3,3,3,64) & (64,)
                name2dst['conv2d'].set_weights([duplicate_kernel_rgb_to_six(W_rgb), b_rgb])
            elif layer.name in name2dst and layer.get_weights():
                name2dst[layer.name].set_weights(layer.get_weights())

    return model


if __name__ == '__main__':

    model = make_fcn_vgg_binary(
        input_shape=(None, None, 6),  # or (None, None, 6)
        stride=8,  # 4, 8, 16, or 32
        use_imagenet_weights=False,  # duplicates first‑layer kernels
        output_mode="pred"
    )

    model.summary()

    # one‑shot inference on a huge image
    # img6 = tf.random.uniform([1, 4096, 4096, 6])
    #
    # p = model(img6, training=False)
    #
    # t = perf_counter()
    # p2 = model(img6, training=False).numpy()  # (1, 512, 512, 1) for stride 8
    #
    # print("Time: ", perf_counter() - t)
    #
    # print(p2.shape)
