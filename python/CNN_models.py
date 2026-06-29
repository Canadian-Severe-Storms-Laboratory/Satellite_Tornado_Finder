
from tensorflow.keras import layers, models, backend as K
import numpy as np
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.utils.layer_utils import get_source_inputs
from keras.models import Model
from keras.layers import Activation, Concatenate, Conv2D, GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D, Input, Dense
from keras.layers import MaxPool2D, BatchNormalization, Lambda, DepthwiseConv2D


def channel_split(x, name=''):
    # equipartition
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 2
    c_hat = Lambda(lambda z: z[:, :, :, 0:ip], name='%s/sp%d_slice' % (name, 0))(x)
    c = Lambda(lambda z: z[:, :, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
    return c_hat, c


def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0,1,2,4,3))
    x = K.reshape(x, [-1, height, width, channels])
    return x


def shuffle_unit(inputs, out_channels, bottleneck_ratio,strides=2,stage=1,block=1):
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        raise ValueError('Only channels last supported')

    prefix = 'stage{}/block{}'.format(stage, block)
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    if strides < 2:
        c_hat, c = channel_split(inputs, '{}/spl'.format(prefix))
        inputs = c

    x = Conv2D(bottleneck_channels, kernel_size=(1,1), strides=1, padding='same', name='{}/1x1conv_1'.format(prefix))(inputs)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_1'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_1'.format(prefix))(x)
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', name='{}/3x3dwconv'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv'.format(prefix))(x)
    x = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1conv_2'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_2'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_2'.format(prefix))(x)

    if strides < 2:
        ret = Concatenate(axis=bn_axis, name='{}/concat_1'.format(prefix))([x, c_hat])
    else:
        s2 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', name='{}/3x3dwconv_2'.format(prefix))(inputs)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv_2'.format(prefix))(s2)
        s2 = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1_conv_3'.format(prefix))(s2)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_3'.format(prefix))(s2)
        s2 = Activation('relu', name='{}/relu_1x1conv_3'.format(prefix))(s2)
        ret = Concatenate(axis=bn_axis, name='{}/concat_2'.format(prefix))([x, s2])

    ret = Lambda(channel_shuffle, name='{}/channel_shuffle'.format(prefix))(ret)

    return ret


def block(x, channel_map, bottleneck_ratio, repeat=1, stage=1):
    x = shuffle_unit(x, out_channels=channel_map[stage-1],
                      strides=2,bottleneck_ratio=bottleneck_ratio,stage=stage,block=1)

    for i in range(1, repeat+1):
        x = shuffle_unit(x, out_channels=channel_map[stage-1],strides=1,
                          bottleneck_ratio=bottleneck_ratio,stage=stage, block=(1+i))

    return x


def _conv_bn_relu(x, filters, kernel_size, strides=1, name=None):
    """Convolution → BatchNorm → ReLU."""
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=None if name is None else f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_bn")(x)
    x = layers.Activation("relu", name=None if name is None else f"{name}_relu")(x)
    return x


def _basic_block(x, filters, stride, downsample, block_name):
    """A ResNet ‘basic block’ with two 3×3 conv layers."""
    identity = x

    # First conv
    x = _conv_bn_relu(x, filters, 3, strides=stride, name=f"{block_name}_conv1")

    # Second conv (no ReLU yet)
    x = layers.Conv2D(
        filters,
        3,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{block_name}_conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{block_name}_bn2")(x)

    # Optional projection to match shape
    if downsample is not None:
        identity = downsample(identity)

    # Add & ReLU
    x = layers.Add(name=f"{block_name}_add")([x, identity])
    x = layers.Activation("relu", name=f"{block_name}_out")(x)
    return x


def _make_layer(x, filters, blocks, stride, stage):
    """
    Build one of the four main ResNet stages.
    `blocks` = number of basic blocks in this stage.
    """
    # Projection shortcut (1×1) for the first block if the shape changes
    downsample = None
    in_channels = K.int_shape(x)[-1]
    if stride != 1 or in_channels != filters:
        downsample = models.Sequential(
            [
                layers.Conv2D(
                    filters,
                    1,
                    strides=stride,
                    use_bias=False,
                    kernel_initializer="he_normal",
                    name=f"conv{stage}_0_proj",
                ),
                layers.BatchNormalization(name=f"bn{stage}_0_proj"),
            ],
            name=f"down{stage}_0",
        )

    x = _basic_block(x, filters, stride, downsample, block_name=f"conv{stage}_0")

    # Remaining blocks
    for i in range(1, blocks):
        x = _basic_block(
            x,
            filters,
            stride=1,
            downsample=None,
            block_name=f"conv{stage}_{i}",
        )
    return x


def ResNet18(input_shape=(224, 224, 3)):
    """
    Build a Keras model of ResNet‑18.
    Args
    ----
    input_shape : tuple, input image shape (H, W, C)
    num_classes : int, number of output classes
    include_top : bool, whether to include the final dense layer
    """
    inputs = layers.Input(shape=input_shape)

    # Stem
    x = _conv_bn_relu(inputs, 64, 7, strides=2, name="conv1")
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name="pool1")(x)

    # 4 stages: (filters, blocks, stride of first block)
    cfg = [(64, 2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2)]
    for stage, (filters, blocks, stride) in enumerate(cfg, start=2):
        x = _make_layer(x, filters, blocks, stride, stage)

    # Head
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)

    return models.Model(inputs, x, name="ResNet18")


def VGG9(input_shape=(224, 224, 3)):
    inp = layers.Input(shape=input_shape)

    # Block 1 — Conv64
    x = layers.Conv2D(64, 3, padding="same", activation="relu",
                      kernel_initializer="he_normal", name="conv1")(inp)
    x = layers.MaxPooling2D(2, 2, name="pool1")(x)

    # Block 2 — Conv128
    x = layers.Conv2D(128, 3, padding="same", activation="relu",
                      kernel_initializer="he_normal", name="conv2")(x)
    x = layers.MaxPooling2D(2, 2, name="pool2")(x)

    # Block 3 — 2 × Conv256
    #for i in range(2):
    x = layers.Conv2D(256, 3, padding="same", activation="relu",
                          kernel_initializer="he_normal",
                          name=f"conv3")(x)
    x = layers.MaxPooling2D(2, 2, name="pool3")(x)

    # Block 4 — 2 × Conv512
    #for i in range(2):
    x = layers.Conv2D(512, 3, padding="same", activation="relu",
                          kernel_initializer="he_normal",
                          name=f"conv4")(x)
    x = layers.MaxPooling2D(2, 2, name="pool4")(x)

    # Block 5 — 2 × Conv512
    #for i in range(2):
    x = layers.Conv2D(512, 3, padding="same", activation="relu",
                          kernel_initializer="he_normal",
                          name=f"conv5")(x)
    x = layers.MaxPooling2D(2, 2, name="pool5")(x)

    # Classifier head
    x = layers.Flatten()(x)

    return models.Model(inp, x, name="VGG9")


def tiny_test():
    inp = layers.Input(shape=(32, 32, 6))
    x = layers.Conv2D(1, 3, padding="same", activation="relu", kernel_initializer="he_normal", name="conv1")(inp)
    x = layers.Flatten()(x)

    return models.Model(inp, x, name="tiny")


def VGG_mini(input_shape=(224, 224, 3)):
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, 3, padding="same", activation="relu", kernel_initializer="he_normal", name="conv1")(inp)
    x = layers.MaxPooling2D(2, 2, name="pool1")(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu",
                      kernel_initializer="he_normal", name="conv2")(x)
    x = layers.MaxPooling2D(2, 2, name="pool2")(x)

    for i in range(2):
        x = layers.Conv2D(64, 3, padding="same", activation="relu",
                          kernel_initializer="he_normal",
                          name=f"conv3_{i + 1}")(x)
    x = layers.MaxPooling2D(2, 2, name="pool3")(x)

    for i in range(2):
        x = layers.Conv2D(128, 3, padding="same", activation="relu",
                          kernel_initializer="he_normal",
                          name=f"conv4_{i + 1}")(x)
    x = layers.MaxPooling2D(2, 2, name="pool4")(x)

    # Classifier head
    x = layers.Flatten()(x)

    return models.Model(inp, x, name="VGG9")


def VGG_micro(input_shape=(224, 224, 3)):
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(8, 3, padding="same", activation="relu", kernel_initializer="he_normal", name="conv1")(inp)
    x = layers.MaxPooling2D(2, 2, name="pool1")(x)

    x = layers.Conv2D(16, 3, padding="same", activation="relu", kernel_initializer="he_normal", name="conv2")(x)
    x = layers.MaxPooling2D(2, 2, name="pool2")(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu", kernel_initializer="he_normal", name=f"conv3")(x)
    x = layers.MaxPooling2D(2, 2, name="pool3")(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu", kernel_initializer="he_normal", name=f"conv4")(x)
    x = layers.MaxPooling2D(2, 2, name="pool4")(x)

    # Classifier head
    x = layers.Flatten()(x)

    return models.Model(inp, x, name="VGG9")


def VGG6(input_shape=(224, 224, 3)):
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, 3, padding="same", activation="relu",
                      kernel_initializer="he_normal", name="conv1")(inp)
    x = layers.MaxPooling2D(2, 2, name="pool1")(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu",
                      kernel_initializer="he_normal", name="conv2")(x)
    x = layers.MaxPooling2D(2, 2, name="pool2")(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu",
                          kernel_initializer="he_normal",
                          name=f"conv3")(x)
    x = layers.MaxPooling2D(2, 2, name="pool3")(x)

    x = layers.Conv2D(512, 3, padding="same", activation="relu",
                          kernel_initializer="he_normal",
                          name=f"conv4")(x)
    x = layers.MaxPooling2D(2, 2, name="pool4")(x)

    x = layers.Conv2D(512, 3, padding="same", activation="relu",
                          kernel_initializer="he_normal",
                          name=f"conv5")(x)
    x = layers.MaxPooling2D(2, 2, name="pool5")(x)

    # Classifier head
    x = layers.Flatten()(x)

    return models.Model(inp, x, name="VGG9")


def ShuffleNetV2(include_top=True,
                 input_tensor=None,
                 scale_factor=1.0,
                 pooling='max',
                 input_shape=(224,224,3),
                 load_model=None,
                 num_shuffle_units=[3,7,3],
                 bottleneck_ratio=1,
                 classes=1000):
    if K.backend() != 'tensorflow':
        raise RuntimeError('Only tensorflow supported for now')
    name = 'ShuffleNetV2_{}_{}_{}'.format(scale_factor, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))
    input_shape = _obtain_input_shape(input_shape, default_size=224, min_size=28, require_flatten=include_top,
                                      data_format=K.image_data_format())
    out_dim_stage_two = {0.5:48, 1:116, 1.5:176, 2:244}

    if pooling not in ['max', 'avg']:
        raise ValueError('Invalid value for pooling')
    if not (float(scale_factor)*4).is_integer():
        raise ValueError('Invalid value for scale_factor, should be x over 4')
    exp = np.insert(np.arange(len(num_shuffle_units), dtype=np.float32), 0, 0)  # [0., 0., 1., 2.]
    out_channels_in_stage = 2**exp
    out_channels_in_stage *= out_dim_stage_two[bottleneck_ratio]  #  calculate output channels for each stage
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # create shufflenet architecture
    x = Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2),
               activation='relu', name='conv1')(img_input)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)

    # create stages containing shufflenet units beginning at stage 2
    for stage in range(len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = block(x, out_channels_in_stage,
                   repeat=repeat,
                   bottleneck_ratio=bottleneck_ratio,
                   stage=stage + 2)

    if bottleneck_ratio < 2:
        k = 1024
    else:
        k = 2048
    x = Conv2D(k, kernel_size=1, padding='same', strides=1, name='1x1conv5_out', activation='relu')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name='global_max_pool')(x)

    if include_top:
        x = Dense(classes, name='fc')(x)
        x = Activation('softmax', name='softmax')(x)

    if input_tensor:
        inputs = get_source_inputs(input_tensor)

    else:
        inputs = img_input

    model = Model(inputs, x, name=name)

    if load_model:
        model.load_weights('', by_name=True)

    return model
