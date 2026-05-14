# -*- coding: utf-8 -*-
import argparse
import json
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

from backbone_utils import BACKBONES, build_binary_model
from tflite_converter import lite_converter

def dataset_dirs(dataset, baseline):
    root = os.path.join("../datasets", dataset)
    if baseline == "E-BAMA":
        return os.path.join(root, "train_ebama_sim"), os.path.join(root, "test_ebama_sim")
    return os.path.join(root, "train_stop_sim"), os.path.join(root, "test_stop_sim")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="GTSRB", choices=["GTSRB", "CIFAR10", "FLOWERS"])
    parser.add_argument("--backbone", default="MobileNetV2", choices=BACKBONES)
    parser.add_argument("--baseline", default="BAMA", choices=["BAMA", "E-BAMA"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=160)
    args = parser.parse_args()

    train_dir, test_dir = dataset_dirs(args.dataset, args.baseline)
    img_size = (args.img_size, args.img_size)
    baseline_token = args.baseline.replace("-", "")
    model_name = "{}_{}_{}_stop_sim".format(args.backbone, args.dataset, baseline_token)

    result_dir = "results_json/{}/{}/feature_extraction/{}".format(args.dataset, args.backbone, args.baseline)
    fig_dir = "figures/{}/{}/feature_extraction/{}".format(args.dataset, args.backbone, args.baseline)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    train_dataset = image_dataset_from_directory(train_dir, shuffle=True, batch_size=args.batch_size, image_size=img_size)
    test_dataset_raw = image_dataset_from_directory(test_dir, shuffle=True, batch_size=args.batch_size, image_size=img_size)

    print("Dataset:", args.dataset)
    print("Backbone:", args.backbone)
    print("Baseline data:", args.baseline)
    print("Train classes:", train_dataset.class_names)
    print("Test classes:", test_dataset_raw.class_names)

    test_batches = tf.data.experimental.cardinality(test_dataset_raw).numpy()
    if test_batches >= 2:
        validation_dataset = test_dataset_raw.take(max(1, test_batches // 2))
        test_dataset = test_dataset_raw.skip(max(1, test_batches // 2))
    else:
        validation_dataset = test_dataset_raw
        test_dataset = test_dataset_raw

    autotune = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=autotune)
    validation_dataset = validation_dataset.prefetch(buffer_size=autotune)
    test_dataset = test_dataset.prefetch(buffer_size=autotune)

    model, base_model = build_binary_model(args.backbone, img_size=img_size, trainable_base=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(train_dataset, epochs=args.epochs, validation_data=validation_dataset)

    with open(os.path.join(result_dir, "{}_history.json".format(model_name)), "w") as f:
        json.dump(history.history, f, indent=2)

    plt.figure()
    plt.plot(history.history.get("loss", []), label="train_loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "{}_loss.png".format(model_name)), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(history.history.get("accuracy", []), label="train_accuracy")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "{}_accuracy.png".format(model_name)), dpi=300)
    plt.close()

    loss, accuracy = model.evaluate(test_dataset)
    metrics = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "baseline_data": args.baseline,
        "training_mode": "feature_extraction",
        "model_name": model_name,
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "class_names": train_dataset.class_names,
    }
    with open(os.path.join(result_dir, "{}_metrics.json".format(model_name)), "w") as f:
        json.dump(metrics, f, indent=2)

    saved_model_path = "fea_ext_bin_models/{}/{}/{}/{}".format(args.dataset, args.backbone, args.baseline, model_name)
    tflite_dir = "tflite_models/fea_ext_bin_models/{}/{}/{}/{}".format(args.dataset, args.backbone, args.baseline, model_name)
    tflite_path = os.path.join(tflite_dir, "{}.tflite".format(model_name))
    os.makedirs(tflite_dir, exist_ok=True)

    model.save(saved_model_path)
    lite_converter(saved_model_path, tflite_path)
    print("Saved model:", saved_model_path)
    print("Saved TFLite:", tflite_path)
    print("Test accuracy:", accuracy)

if __name__ == "__main__":
    main()
