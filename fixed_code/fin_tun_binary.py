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
        return os.path.join(root, "train_ebama_sim"), os.path.join(root, "valid_sim"), os.path.join(root, "test_ebama_sim")
    return os.path.join(root, "train_sim"), os.path.join(root, "valid_sim"), os.path.join(root, "test_sim")

def parse_layers(value):
    if value == "all":
        return [10, 20, 30, 40, 50, 60]
    return [int(x.strip()) for x in value.split(",") if x.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="GTSRB", choices=["GTSRB", "CIFAR10", "FLOWERS"])
    parser.add_argument("--backbone", default="MobileNetV2", choices=BACKBONES)
    parser.add_argument("--baseline", default="BAMA", choices=["BAMA", "E-BAMA"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine_epochs", type=int, default=10)
    parser.add_argument("--fine_layers", default="60", help="'60', '10,20,60', or 'all'")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=160)
    args = parser.parse_args()

    train_dir, valid_dir, test_dir = dataset_dirs(args.dataset, args.baseline)
    img_size = (args.img_size, args.img_size)
    fine_layers_list = parse_layers(args.fine_layers)

    for fine_layers in fine_layers_list:
        print("=" * 80)
        print("Dataset:", args.dataset)
        print("Backbone:", args.backbone)
        print("Baseline data:", args.baseline)
        print("Fine-tune last layers:", fine_layers)

        train_dataset = image_dataset_from_directory(train_dir, shuffle=True, batch_size=args.batch_size, image_size=img_size)
        validation_dataset = image_dataset_from_directory(valid_dir, shuffle=True, batch_size=args.batch_size, image_size=img_size)
        test_dataset = image_dataset_from_directory(test_dir, shuffle=True, batch_size=args.batch_size, image_size=img_size)

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

        loss0, acc0 = model.evaluate(validation_dataset)
        history = model.fit(train_dataset, epochs=args.epochs, validation_data=validation_dataset)

        base_model.trainable = True
        fine_tune_at = max(0, len(base_model.layers) - fine_layers)
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        actual_trainable = sum(1 for layer in base_model.layers if layer.trainable)
        final_name = "{}_{}_{}_{}_sim".format(args.backbone, args.dataset, args.baseline.replace("-", ""), actual_trainable)

        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        total_epochs = args.epochs + args.fine_epochs
        initial_epoch = history.epoch[-1] + 1 if history.epoch else args.epochs
        history_fine = model.fit(train_dataset, epochs=total_epochs, initial_epoch=initial_epoch, validation_data=validation_dataset)

        history_all = {}
        for k, v in history.history.items():
            history_all[k] = list(v)
        for k, v in history_fine.history.items():
            history_all.setdefault(k, [])
            history_all[k].extend(list(v))

        result_dir = "results_json/{}/{}/fine_tuning/{}".format(args.dataset, args.backbone, args.baseline)
        fig_dir = "figures/{}/{}/fine_tuning/{}".format(args.dataset, args.backbone, args.baseline)
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)

        with open(os.path.join(result_dir, "{}_history.json".format(final_name)), "w") as f:
            json.dump(history_all, f, indent=2)

        plt.figure()
        plt.plot(history_all.get("loss", []), label="train_loss")
        if "val_loss" in history_all:
            plt.plot(history_all["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(fig_dir, "{}_loss.png".format(final_name)), dpi=300)
        plt.close()

        plt.figure()
        plt.plot(history_all.get("accuracy", []), label="train_accuracy")
        if "val_accuracy" in history_all:
            plt.plot(history_all["val_accuracy"], label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(fig_dir, "{}_accuracy.png".format(final_name)), dpi=300)
        plt.close()

        loss, accuracy = model.evaluate(test_dataset)
        metrics = {
            "dataset": args.dataset,
            "backbone": args.backbone,
            "baseline_data": args.baseline,
            "training_mode": "fine_tuning",
            "fine_layers_requested": fine_layers,
            "fine_layers_actual_trainable": actual_trainable,
            "model_name": final_name,
            "initial_loss": float(loss0),
            "initial_accuracy": float(acc0),
            "test_loss": float(loss),
            "test_accuracy": float(accuracy),
            "initial_epochs": args.epochs,
            "fine_epochs": args.fine_epochs,
            "batch_size": args.batch_size,
            "img_size": args.img_size,
            "class_names": train_dataset.class_names,
        }
        with open(os.path.join(result_dir, "{}_metrics.json".format(final_name)), "w") as f:
            json.dump(metrics, f, indent=2)

        saved_model_path = "fin_tun_bin_models/{}/{}/{}/{}".format(args.dataset, args.backbone, args.baseline, final_name)
        tflite_dir = "tflite_models/fin_tun_bin_models/{}/{}/{}/{}".format(args.dataset, args.backbone, args.baseline, final_name)
        tflite_path = os.path.join(tflite_dir, "{}.tflite".format(final_name))
        os.makedirs(tflite_dir, exist_ok=True)

        model.save(saved_model_path)
        lite_converter(saved_model_path, tflite_path)
        print("Saved model:", saved_model_path)
        print("Saved TFLite:", tflite_path)
        print("Test accuracy:", accuracy)

if __name__ == "__main__":
    main()
