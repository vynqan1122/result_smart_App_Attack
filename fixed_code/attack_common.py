# -*- coding: utf-8 -*-
import argparse
import json
import os
import pathlib

import eagerpy as ep
import foolbox as fb
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from backbone_utils import BACKBONES, build_base_model

def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except Exception:
            pass

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def find_saved_model(dataset, backbone, baseline, model_name):
    candidates = [
        "exp_models/{}/{}".format(dataset, model_name),
        "fea_ext_bin_models/{}/{}/{}/{}".format(dataset, backbone, baseline, model_name),
        "fin_tun_bin_models/{}/{}/{}/{}".format(dataset, backbone, baseline, model_name),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Cannot find SavedModel. Tried:\\n" + "\\n".join(candidates))

def find_tflite_model(dataset, backbone, baseline, model_name):
    candidates = [
        "exp_models/{}/{}.tflite".format(dataset, model_name),
        "tflite_models/fea_ext_bin_models/{}/{}/{}/{}/{}.tflite".format(dataset, backbone, baseline, model_name, model_name),
        "tflite_models/fin_tun_bin_models/{}/{}/{}/{}/{}.tflite".format(dataset, backbone, baseline, model_name, model_name),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Cannot find TFLite model. Tried:\\n" + "\\n".join(candidates))

def load_interpreter(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_for_tflite(img_path, input_shape):
    # Important: do NOT preprocess twice. The model graph already contains preprocess_input.
    _, h, w, _ = input_shape
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(h, w))
    arr = tf.keras.preprocessing.image.img_to_array(img).astype(np.float32)
    arr = tf.expand_dims(arr, 0)
    return arr

def predict_file(interpreter, img_path):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    x = preprocess_for_tflite(img_path, input_details[0]["shape"])
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(output_details[0]["index"])
    return int(np.argmax(y[0]))

def load_images(image_dir, interpreter, num_images, target_label=1):
    files = sorted(pathlib.Path(image_dir).glob("*"))
    files = [f for f in files if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".ppm", ".bmp"]]

    images, labels, ori_labels = [], [], {}
    skipped = 0
    for f in files:
        pred = predict_file(interpreter, str(f))
        if pred != target_label:
            skipped += 1
            print(f, pred, "SKIP")
            continue

        img = tf.keras.preprocessing.image.load_img(str(f), target_size=(160, 160))
        arr = tf.keras.preprocessing.image.img_to_array(img).astype(np.float32)
        images.append(arr)
        labels.append(pred)
        ori_labels[f.name] = pred
        print(f, pred, "USE")

        if len(images) >= num_images:
            break

    if len(images) == 0:
        raise RuntimeError("No valid target images found in {}. TFLite did not predict label {} for any image.".format(image_dir, target_label))

    print("Used target images:", len(images))
    print("Skipped images:", skipped)
    return np.stack(images), np.array(labels, dtype=np.int64), ori_labels

def save_advs(advs_list, dataset, backbone, model_name, baseline, attack_algo, round_id):
    save_root = "adv_examples/{}/{}/{}/{}/{}/round_{}".format(dataset, backbone, model_name, baseline, attack_algo, round_id)
    ensure_dir(save_root)
    saved = []
    for advs in advs_list:
        for i, adv in enumerate(advs):
            arr = adv.numpy()
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            out_path = "{}/adv_{:03d}_{:05d}.png".format(save_root, round_id, i)
            Image.fromarray(arr).save(out_path)
            saved.append(out_path)
    return save_root, saved

def infer_folder(interpreter, folder):
    results = {}
    files = sorted(pathlib.Path(folder).glob("*"))
    files = [f for f in files if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".ppm", ".bmp"]]
    for f in files:
        pred = predict_file(interpreter, str(f))
        results[f.name] = pred
        print(f, pred)
    return results

def calc_success(ori_labels, adv_labels, target_label=1):
    # File names differ, so compare by order.
    ori_values = list(ori_labels.values())
    adv_values = list(adv_labels.values())
    n = min(len(ori_values), len(adv_values))
    if n == 0:
        return 0.0
    total, success = 0, 0
    for i in range(n):
        if ori_values[i] == target_label:
            total += 1
            if adv_values[i] != target_label:
                success += 1
    if total == 0:
        return 0.0
    return success / total

def build_attack(attack_algo):
    if attack_algo == "FGSM":
        return fb.attacks.FGSM(), "FGSM"
    if attack_algo == "CW":
        return fb.attacks.L2CarliniWagnerAttack(steps=40), "CW"
    if attack_algo == "CAN":
        return fb.attacks.L2ClippingAwareAdditiveGaussianNoiseAttack(), "CAN"
    raise ValueError("Unknown attack_algo: {}".format(attack_algo))

def default_epsilon(attack_algo):
    # Bounds are 0..255. Paper FGSM epsilon 0.025 approximately equals 6.0 here.
    if attack_algo == "FGSM":
        return 6.0
    if attack_algo == "CW":
        return 20.0
    if attack_algo == "CAN":
        return 20.0
    return 20.0

def load_source_model(baseline, backbone, saved_model_path):
    if baseline in ["BAMA", "E-BAMA"]:
        model = tf.keras.models.load_model(saved_model_path)
        model.trainable = False
        return model

    # PMA approximation: use ImageNet pre-trained backbone with a binary head.
    # This keeps the PMA baseline executable, but BAMA/E-BAMA are the main faithful reproduction targets.
    base = build_base_model(backbone, input_shape=(160, 160, 3), include_top=False)
    base.trainable = False
    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(2)(x)
    model = tf.keras.Model(inputs, outputs)
    model.trainable = False
    return model

def run_attack(baseline):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="GTSRB", choices=["GTSRB", "CIFAR10", "FLOWERS"])
    parser.add_argument("--backbone", default="MobileNetV2", choices=BACKBONES)
    parser.add_argument("--baseline_data", default=None, choices=[None, "BAMA", "E-BAMA"])
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--attack_algo", default="FGSM", choices=["FGSM", "CW", "CAN"])
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--num_images", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=None)
    args = parser.parse_args()

    setup_gpu()

    baseline_data = args.baseline_data or ("E-BAMA" if baseline == "E-BAMA" else "BAMA")
    model_name = args.model_name or "{}_{}_{}_stop_sim".format(args.backbone, args.dataset, baseline_data.replace("-", ""))
    epsilon = args.epsilon if args.epsilon is not None else default_epsilon(args.attack_algo)

    image_dir = "exp_models/{}/stop".format(args.dataset)
    tflite_path = find_tflite_model(args.dataset, args.backbone, baseline_data, model_name)
    saved_model_path = find_saved_model(args.dataset, args.backbone, baseline_data, model_name)

    ensure_dir("logs/{}/{}/{}/{}".format(args.dataset, args.backbone, baseline_data, baseline))
    ensure_dir("results_json/{}/{}/attacks/{}/{}".format(args.dataset, args.backbone, baseline_data, baseline))
    ensure_dir("adv_examples/{}/{}/{}/{}".format(args.dataset, args.backbone, model_name, baseline))

    print("=" * 80)
    print("Dataset:", args.dataset)
    print("Backbone:", args.backbone)
    print("Baseline:", baseline)
    print("Baseline data:", baseline_data)
    print("Attack algorithm:", args.attack_algo)
    print("Model:", model_name)
    print("Epsilon:", epsilon)
    print("TFLite:", tflite_path)
    print("SavedModel:", saved_model_path)
    print("Images:", image_dir)
    print("=" * 80)

    interpreter = load_interpreter(tflite_path)
    images_np, labels_np, ori_labels = load_images(image_dir, interpreter, args.num_images, target_label=1)

    source_model = load_source_model(baseline, args.backbone, saved_model_path)
    fmodel = fb.TensorFlowModel(source_model, bounds=(0, 255))
    attack, attack_name = build_attack(args.attack_algo)

    images = ep.astensor(tf.convert_to_tensor(images_np))
    labels = ep.astensor(tf.convert_to_tensor(labels_np))

    success_rates = []
    for r in tqdm(range(args.rounds)):
        raw, advs_list, success = attack(fmodel, images, labels, epsilons=[epsilon])
        adv_dir, saved = save_advs(advs_list, args.dataset, args.backbone, model_name, baseline, args.attack_algo, r)
        adv_labels = infer_folder(interpreter, adv_dir)
        rate = calc_success(ori_labels, adv_labels, target_label=1)
        success_rates.append(rate)
        print("Foolbox internal success:", success.float32().mean().item())
        print("TFLite attack success rate:", rate)

    result = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "baseline": baseline,
        "baseline_data": baseline_data,
        "attack_algo": args.attack_algo,
        "attack_name": attack_name,
        "model_name": model_name,
        "epsilon": epsilon,
        "rounds": args.rounds,
        "num_images": args.num_images,
        "success_rates": success_rates,
        "min_success_rate": float(min(success_rates)),
        "max_success_rate": float(max(success_rates)),
        "mean_success_rate": float(np.mean(success_rates)),
        "note": "PMA is executable approximation; BAMA/E-BAMA are the main binary-model attacks."
    }

    out_json = "results_json/{}/{}/attacks/{}/{}/{}_{}_{}_results.json".format(
        args.dataset, args.backbone, baseline_data, baseline, baseline, args.attack_algo, model_name
    )
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print("Saved result:", out_json)
