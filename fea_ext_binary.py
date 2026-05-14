# -*- coding: utf-8 -*-
from tflite_converter import lite_converter
from backbone_utils import BACKBONES, build_binary_model, compile_binary
import os, json, argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory

def ensure(path): os.makedirs(path, exist_ok=True)

def plot_history(history, out_prefix):
    ensure(os.path.dirname(out_prefix)); h = history.history
    plt.figure(); plt.plot(h.get("loss", []), label="train_loss")
    if "val_loss" in h: plt.plot(h["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True); plt.savefig(out_prefix + "_loss.png", dpi=200); plt.close()
    plt.figure(); plt.plot(h.get("accuracy", []), label="train_accuracy")
    if "val_accuracy" in h: plt.plot(h["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True); plt.savefig(out_prefix + "_accuracy.png", dpi=200); plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["GTSRB", "CIFAR10", "FLOWERS"])
    parser.add_argument("--backbone", default="MobileNetV2", choices=BACKBONES)
    parser.add_argument("--baseline", default="BAMA", choices=["BAMA", "E-BAMA"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    dataset_name, backbone_name, baseline = args.dataset, args.backbone, args.baseline
    path = os.path.join("../datasets", dataset_name)
    train_sub = "train_ebama_sim" if baseline == "E-BAMA" else "train_stop_sim"
    test_sub = "test_ebama_sim" if baseline == "E-BAMA" else "test_stop_sim"
    train_dir, test_dir = os.path.join(path, train_sub), os.path.join(path, test_sub)
    img_size = (160,160); model_name = f"{backbone_name}_{dataset_name}_{baseline}_stop_sim"
    ensure(f"results_json/{dataset_name}/{backbone_name}/feature_extraction/{baseline}"); ensure(f"figures/{dataset_name}/{backbone_name}/feature_extraction/{baseline}")
    train_ds_raw = image_dataset_from_directory(train_dir, shuffle=True, batch_size=args.batch_size, image_size=img_size)
    test_ds_raw = image_dataset_from_directory(test_dir, shuffle=True, batch_size=args.batch_size, image_size=img_size)
    class_names = list(train_ds_raw.class_names); test_class_names = list(test_ds_raw.class_names)
    print("Dataset:", dataset_name, "Backbone:", backbone_name, "Baseline data:", baseline)
    print("Train classes:", class_names); print("Test classes:", test_class_names)
    test_batches = int(tf.data.experimental.cardinality(test_ds_raw).numpy())
    if test_batches >= 2:
        valid_ds_raw = test_ds_raw.take(max(1, test_batches // 2)); test_ds = test_ds_raw.skip(max(1, test_batches // 2))
    else:
        valid_ds_raw = test_ds_raw; test_ds = test_ds_raw
    autotune = tf.data.experimental.AUTOTUNE
    train_ds = train_ds_raw.prefetch(buffer_size=autotune); valid_ds = valid_ds_raw.prefetch(buffer_size=autotune); test_ds = test_ds.prefetch(buffer_size=autotune)
    model, _ = build_binary_model(backbone_name, img_size=img_size, train_backbone=False); compile_binary(model, lr=1e-3)
    history = model.fit(train_ds, epochs=args.epochs, validation_data=valid_ds)
    loss, acc = model.evaluate(test_ds)
    result_dir = f"results_json/{dataset_name}/{backbone_name}/feature_extraction/{baseline}"
    with open(f"{result_dir}/{model_name}_history.json", "w") as f: json.dump(history.history, f, indent=2)
    metrics = {"dataset": dataset_name, "backbone": backbone_name, "baseline": baseline, "training_mode": "feature_extraction", "model_name": model_name, "class_names": class_names, "test_loss": float(loss), "test_accuracy": float(acc), "epochs": args.epochs}
    with open(f"{result_dir}/{model_name}_metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    plot_history(history, f"figures/{dataset_name}/{backbone_name}/feature_extraction/{baseline}/{model_name}")
    saved_model_path = f"fea_ext_bin_models/{dataset_name}/{backbone_name}/{baseline}/{model_name}"
    tflite_dir = f"tflite_models/fea_ext_bin_models/{dataset_name}/{backbone_name}/{baseline}/{model_name}"; ensure(tflite_dir)
    model.save(saved_model_path); lite_converter(saved_model_path, os.path.join(tflite_dir, model_name + ".tflite"))
    print("Saved model:", saved_model_path); print("Saved TFLite:", os.path.join(tflite_dir, model_name + ".tflite"))
if __name__ == "__main__": main()
