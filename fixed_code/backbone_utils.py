# -*- coding: utf-8 -*-
import tensorflow as tf

BACKBONES = ["MobileNetV2", "InceptionV3", "ResNet50V2"]

def get_preprocess(backbone):
    if backbone == "MobileNetV2":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    if backbone == "InceptionV3":
        return tf.keras.applications.inception_v3.preprocess_input
    if backbone == "ResNet50V2":
        return tf.keras.applications.resnet_v2.preprocess_input
    raise ValueError("Unknown backbone: {}".format(backbone))

def build_base_model(backbone, input_shape=(160, 160, 3), include_top=False):
    if backbone == "MobileNetV2":
        return tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=include_top, weights="imagenet")
    if backbone == "InceptionV3":
        return tf.keras.applications.InceptionV3(input_shape=input_shape, include_top=include_top, weights="imagenet")
    if backbone == "ResNet50V2":
        return tf.keras.applications.ResNet50V2(input_shape=input_shape, include_top=include_top, weights="imagenet")
    raise ValueError("Unknown backbone: {}".format(backbone))

def build_binary_model(backbone, img_size=(160, 160), trainable_base=False):
    """Binary transfer-learning model. Input is raw pixels 0..255."""
    img_shape = img_size + (3,)
    preprocess_input = get_preprocess(backbone)
    base_model = build_base_model(backbone, input_shape=img_shape, include_top=False)
    base_model.trainable = trainable_base

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    inputs = tf.keras.Input(shape=img_shape)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(2)(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base_model
