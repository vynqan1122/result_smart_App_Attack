# -*- coding: utf-8 -*-
import tensorflow as tf
BACKBONES = ["MobileNetV2", "InceptionV3", "ResNet50V2"]
FINE_TUNE_NUMBERS = {"MobileNetV2": [10,20,30,40,50,60], "InceptionV3": [10,20,30,40,50,60], "ResNet50V2": [10,20,30,40,50,60]}

def get_backbone(backbone_name, input_shape=(160,160,3), include_top=False):
    if backbone_name == "MobileNetV2":
        return tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=include_top, weights="imagenet")
    if backbone_name == "InceptionV3":
        return tf.keras.applications.InceptionV3(input_shape=input_shape, include_top=include_top, weights="imagenet")
    if backbone_name == "ResNet50V2":
        return tf.keras.applications.ResNet50V2(input_shape=input_shape, include_top=include_top, weights="imagenet")
    raise ValueError(f"Unknown backbone: {backbone_name}")

def get_preprocess(backbone_name):
    if backbone_name == "MobileNetV2": return tf.keras.applications.mobilenet_v2.preprocess_input
    if backbone_name == "InceptionV3": return tf.keras.applications.inception_v3.preprocess_input
    if backbone_name == "ResNet50V2": return tf.keras.applications.resnet_v2.preprocess_input
    raise ValueError(f"Unknown backbone: {backbone_name}")

def build_binary_model(backbone_name, img_size=(160,160), train_backbone=False):
    preprocess_input = get_preprocess(backbone_name)
    img_shape = img_size + (3,)
    base_model = get_backbone(backbone_name, input_shape=img_shape, include_top=False)
    base_model.trainable = bool(train_backbone)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ], name="data_augmentation")
    inputs = tf.keras.Input(shape=img_shape, name="raw_image_input")
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(2, name="binary_logits")(x)
    return tf.keras.Model(inputs, outputs, name=f"{backbone_name}_binary_model"), base_model

def compile_binary(model, lr=1e-3, optimizer="adam"):
    opt = tf.keras.optimizers.RMSprop(learning_rate=lr) if optimizer == "rmsprop" else tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    return model
