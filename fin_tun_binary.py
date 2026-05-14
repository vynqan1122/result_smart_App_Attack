# -*- coding: utf-8 -*-
from tflite_converter import lite_converter
from backbone_utils import BACKBONES, FINE_TUNE_NUMBERS, build_binary_model, compile_binary
import os, json, argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory

def ensure(path): os.makedirs(path, exist_ok=True)

def plot_history(hist, out_prefix):
    ensure(os.path.dirname(out_prefix))
    plt.figure(); plt.plot(hist.get("loss", []), label="train_loss")
    if "val_loss" in hist: plt.plot(hist["val_loss"], label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True); plt.savefig(out_prefix + "_loss.png", dpi=200); plt.close()
    plt.figure(); plt.plot(hist.get("accuracy", []), label="train_accuracy")
    if "val_accuracy" in hist: plt.plot(hist["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True); plt.savefig(out_prefix + "_accuracy.png", dpi=200); plt.close()

def merge_history(h1, h2):
    out = {k: list(v) for k, v in h1.history.items()}
    for k, v in h2.history.items(): out.setdefault(k, []).extend(list(v))
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["GTSRB", "CIFAR10", "FLOWERS"])
    parser.add_argument("--backbone", default="MobileNetV2", choices=BACKBONES)
    parser.add_argument("--baseline", default="BAMA", choices=["BAMA", "E-BAMA"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--fine_epochs", type=int, default=1)
    parser.add_argument("--fine_layers", default="60", help="comma-separated fine-tune layer counts, e.g. 10,20,60 or all")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    dataset_name, backbone_name, baseline = args.dataset, args.backbone, args.baseline
    path = os.path.join("../datasets", dataset_name)
    train_sub = "train_ebama_sim" if baseline == "E-BAMA" else "train_sim"
    test_sub = "test_ebama_sim" if baseline == "E-BAMA" else "test_sim"
    valid_sub = "valid_ebama_sim" if baseline == "E-BAMA" else "valid_sim"
    train_dir, test_dir, valid_dir = os.path.join(path, train_sub), os.path.join(path, test_sub), os.path.join(path, valid_sub)
    if not os.path.isdir(valid_dir): valid_dir = os.path.join(path, "valid_sim")
    img_size = (160,160)
    fine_layer_counts = FINE_TUNE_NUMBERS[backbone_name] if args.fine_layers == "all" else [int(x.strip()) for x in args.fine_layers.split(",") if x.strip()]
    for requested_layers in fine_layer_counts:
        print("= " * 40); print("Dataset:", dataset_name, "Backbone:", backbone_name, "Baseline data:", baseline, "Fine layers request:", requested_layers)
        train_ds_raw = image_dataset_from_directory(train_dir, shuffle=True, batch_size=args.batch_size, image_size=img_size)
        valid_ds_raw = image_dataset_from_directory(valid_dir, shuffle=True, batch_size=args.batch_size, image_size=img_size)
        test_ds_raw = image_dataset_from_directory(test_dir, shuffle=True, batch_size=args.batch_size, image_size=img_size)
        class_names = list(train_ds_raw.class_names)
        autotune = tf.data.experimental.AUTOTUNE
        train_ds = train_ds_raw.prefetch(buffer_size=autotune); valid_ds = valid_ds_raw.prefetch(buffer_size=autotune); test_ds = test_ds_raw.prefetch(buffer_size=autotune)
        model, base_model = build_binary_model(backbone_name, img_size=img_size, train_backbone=False); compile_binary(model, lr=1e-3)
        h1 = model.fit(train_ds, epochs=args.epochs, validation_data=valid_ds)
        base_model.trainable = True; total_base_layers = len(base_model.layers); fine_tune_at = max(0, total_base_layers - requested_layers)
        for layer in base_model.layers[:fine_tune_at]: layer.trainable = False
        actual_trainable = sum(1 for layer in base_model.layers if layer.trainable)
        compile_binary(model, lr=1e-4, optimizer="rmsprop")
        h2 = model.fit(train_ds, epochs=args.epochs + args.fine_epochs, initial_epoch=args.epochs, validation_data=valid_ds)
        hist = merge_history(h1, h2); loss, acc = model.evaluate(test_ds)
        model_name = f"{backbone_name}_{dataset_name}_{baseline}_{actual_trainable}_sim"
        result_dir = f"results_json/{dataset_name}/{backbone_name}/fine_tuning/{baseline}/{actual_trainable}"; fig_dir = f"figures/{dataset_name}/{backbone_name}/fine_tuning/{baseline}/{actual_trainable}"
        ensure(result_dir); ensure(fig_dir)
        with open(f"{result_dir}/{model_name}_history.json", "w") as f: json.dump(hist, f, indent=2)
        metrics = {"dataset": dataset_name, "backbone": backbone_name, "baseline": baseline, "training_mode": "fine_tuning", "model_name": model_name, "class_names": class_names, "requested_fine_layers": requested_layers, "actual_trainable_base_layers": actual_trainable, "test_loss": float(loss), "test_accuracy": float(acc), "initial_epochs": args.epochs, "fine_epochs": args.fine_epochs}
        with open(f"{result_dir}/{model_name}_metrics.json", "w") as f: json.dump(metrics, f, indent=2)
        plot_history(hist, f"{fig_dir}/{model_name}")
        saved_model_path = f"fin_tun_bin_models/{dataset_name}/{backbone_name}/{baseline}/{model_name}"; tflite_dir = f"tflite_models/fin_tun_bin_models/{dataset_name}/{backbone_name}/{baseline}/{model_name}"
        ensure(tflite_dir); model.save(saved_model_path); lite_converter(saved_model_path, os.path.join(tflite_dir, model_name + ".tflite"))
        print("Saved model:", saved_model_path); print("Saved TFLite:", os.path.join(tflite_dir, model_name + ".tflite"))
if __name__ == "__main__": main()
