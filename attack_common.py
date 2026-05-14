# -*- coding: utf-8 -*-
import os, json, argparse, pathlib
import numpy as np
import tensorflow as tf
import foolbox as fb
import eagerpy as ep
from PIL import Image
from tqdm import tqdm
from backbone_utils import BACKBONES, get_backbone, get_preprocess


def ensure(path):
    os.makedirs(path, exist_ok=True)


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except Exception:
            pass


def find_tflite(dataset, model_name):
    candidates = ["exp_models/{}/{}.tflite".format(dataset, model_name)]
    candidates += [str(p) for p in pathlib.Path("tflite_models").glob("**/{}.tflite".format(model_name))]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Cannot find TFLite model for {}/{}".format(dataset, model_name))


def find_saved_model(dataset, model_name):
    candidates = ["exp_models/{}/{}".format(dataset, model_name)]
    for root in ["fea_ext_bin_models", "fin_tun_bin_models"]:
        candidates += [str(p) for p in pathlib.Path(root).glob("**/{}".format(model_name))]
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError("Cannot find SavedModel for {}/{}".format(dataset, model_name))


def load_interpreter(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_for_tflite(img_path, input_shape):
    _, h, w, _ = input_shape
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(h, w))
    arr = tf.keras.preprocessing.image.img_to_array(img).astype(np.float32)
    return tf.expand_dims(arr, 0)


def predict_file(interpreter, img_path):
    inp, out = interpreter.get_input_details(), interpreter.get_output_details()
    x = preprocess_for_tflite(img_path, inp[0]["shape"])
    interpreter.set_tensor(inp[0]["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(out[0]["index"])
    return int(np.argmax(y[0]))


def load_target_images(image_dir, interpreter, num_images, target_label=1):
    files = sorted(pathlib.Path(image_dir).glob("*"))
    files = [f for f in files if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".ppm", ".bmp"]]
    images, victim_labels, names = [], [], []
    skipped = 0
    for f in files:
        pred = predict_file(interpreter, str(f))
        if pred != target_label:
            skipped += 1
            continue
        img = tf.keras.preprocessing.image.load_img(str(f), target_size=(160, 160))
        arr = tf.keras.preprocessing.image.img_to_array(img).astype(np.float32)
        images.append(arr)
        victim_labels.append(pred)
        names.append(f.name)
        print(str(f), pred, "USE")
        if len(images) >= num_images:
            break
    if len(images) == 0:
        raise RuntimeError("No valid target images predicted as label={} in {}; skipped={}".format(target_label, image_dir, skipped))
    print("Used images:", len(images), "Skipped:", skipped)
    return np.stack(images).astype(np.float32), np.array(victim_labels, dtype=np.int64), names


def build_source_model(baseline, backbone, saved_model_path):
    if baseline in ["BAMA", "E-BAMA"]:
        model = tf.keras.models.load_model(saved_model_path)
        model.trainable = False
        return model
    preprocess = get_preprocess(backbone)
    base = get_backbone(backbone, input_shape=(160, 160, 3), include_top=True)
    base.trainable = False
    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = preprocess(inputs)
    outputs = base(x, training=False)
    return tf.keras.Model(inputs, outputs, name="{}_PMA_source".format(backbone))


def build_attack(attack_algo, steps=10):
    algo = attack_algo.upper()
    if algo == "FGSM":
        return fb.attacks.FGSM(), "FGSM"
    if algo == "CW":
        return fb.attacks.L2CarliniWagnerAttack(steps=40), "CW"
    if algo == "CAN":
        return fb.attacks.L2ClippingAwareAdditiveGaussianNoiseAttack(), "CAN"
    if algo == "PGD":
        return None, "PGD"
    if algo in ["MIFGSM", "MI-FGSM"]:
        return None, "MIFGSM"
    raise ValueError("Unknown attack_algo: {}".format(attack_algo))


def _loss_for_untargeted_attack(model, x, y):
    logits = model(x, training=False)
    # Binary BAMA/E-BAMA models use logits. This is the main path for this project.
    return tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)


def pgd_attack_tf(model, images_tf, labels_tf, epsilon, alpha, steps, random_start=True):
    x0 = tf.cast(images_tf, tf.float32)
    y = tf.cast(labels_tf, tf.int64)
    eps = tf.cast(epsilon, tf.float32)
    step_alpha = tf.cast(alpha, tf.float32)

    if random_start:
        x_adv = x0 + tf.random.uniform(tf.shape(x0), minval=-eps, maxval=eps, dtype=tf.float32)
        x_adv = tf.clip_by_value(x_adv, 0.0, 255.0)
    else:
        x_adv = tf.identity(x0)

    for _ in range(int(steps)):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            loss = tf.reduce_mean(_loss_for_untargeted_attack(model, x_adv, y))
        grad = tape.gradient(loss, x_adv)
        x_adv = x_adv + step_alpha * tf.sign(grad)
        x_adv = tf.minimum(tf.maximum(x_adv, x0 - eps), x0 + eps)
        x_adv = tf.clip_by_value(x_adv, 0.0, 255.0)

    preds = tf.argmax(model(x_adv, training=False), axis=1, output_type=tf.int64)
    success = tf.not_equal(preds, y)
    return x_adv, success


def mifgsm_attack_tf(model, images_tf, labels_tf, epsilon, alpha, steps, decay=1.0):
    x0 = tf.cast(images_tf, tf.float32)
    y = tf.cast(labels_tf, tf.int64)
    eps = tf.cast(epsilon, tf.float32)
    step_alpha = tf.cast(alpha, tf.float32)
    momentum = tf.zeros_like(x0)
    x_adv = tf.identity(x0)

    for _ in range(int(steps)):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            loss = tf.reduce_mean(_loss_for_untargeted_attack(model, x_adv, y))
        grad = tape.gradient(loss, x_adv)
        grad_norm = tf.reduce_mean(tf.abs(grad), axis=[1, 2, 3], keepdims=True)
        grad = grad / (grad_norm + 1e-12)
        momentum = decay * momentum + grad
        x_adv = x_adv + step_alpha * tf.sign(momentum)
        x_adv = tf.minimum(tf.maximum(x_adv, x0 - eps), x0 + eps)
        x_adv = tf.clip_by_value(x_adv, 0.0, 255.0)

    preds = tf.argmax(model(x_adv, training=False), axis=1, output_type=tf.int64)
    success = tf.not_equal(preds, y)
    return x_adv, success


def save_advs(advs_list, dataset, backbone, training_mode, baseline, attack_algo, model_name, round_id):
    save_root = "adv_examples/{}/{}/{}/{}/{}/{}/round_{}".format(dataset, backbone, training_mode, baseline, attack_algo, model_name, round_id)
    ensure(save_root)
    saved = []
    for advs in advs_list:
        arrs = advs.numpy() if hasattr(advs, "numpy") else np.asarray(advs)
        for i, arr in enumerate(arrs):
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            out_path = "{}/adv_{:03d}_{:05d}.png".format(save_root, round_id, i)
            Image.fromarray(arr).save(out_path)
            saved.append(out_path)
    return save_root, saved


def infer_folder(interpreter, folder):
    labels = []
    for f in sorted(pathlib.Path(folder).glob("*")):
        if f.suffix.lower() not in [".png", ".jpg", ".jpeg", ".ppm", ".bmp"]:
            continue
        pred = predict_file(interpreter, str(f))
        labels.append(pred)
        print(str(f), pred)
    return labels


def calc_asr(original_labels, adv_labels, target_label=1):
    n = min(len(original_labels), len(adv_labels))
    if n == 0:
        return 0.0
    total, success = 0, 0
    for i in range(n):
        if int(original_labels[i]) == target_label:
            total += 1
            if int(adv_labels[i]) != target_label:
                success += 1
    return 0.0 if total == 0 else success / total


def run_attack(baseline):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["GTSRB", "CIFAR10", "FLOWERS"])
    parser.add_argument("--backbone", default="MobileNetV2", choices=BACKBONES)
    parser.add_argument("--training_mode", default="feature_extraction", choices=["feature_extraction", "fine_tuning"])
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--attack_algo", default="CAN", choices=["FGSM", "CW", "CAN", "PGD", "MIFGSM", "MI-FGSM"])
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument("--random_start", action="store_true", help="Use random start for PGD")
    parser.add_argument("--target_label", type=int, default=1)
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--attack_batch_size", type=int, default=8, help="Mini-batch size for PGD/MIFGSM generation to avoid GPU OOM")
    args = parser.parse_args()
    setup_gpu()

    algo = args.attack_algo.upper()
    if algo == "MI-FGSM":
        algo = "MIFGSM"

    epsilon_defaults = {"FGSM": 8.0, "CW": 20.0, "CAN": 20.0, "PGD": 8.0, "MIFGSM": 8.0}
    epsilon = args.epsilon if args.epsilon is not None else epsilon_defaults[algo]
    alpha_defaults = {"PGD": max(epsilon / 4.0, 1.0), "MIFGSM": max(epsilon / 10.0, 1.0)}
    alpha = args.alpha if args.alpha is not None else alpha_defaults.get(algo, None)

    image_dir = args.image_dir if args.image_dir is not None else "exp_models/{}/stop".format(args.dataset)
    tflite_path = find_tflite(args.dataset, args.model_name)
    saved_model_path = find_saved_model(args.dataset, args.model_name)

    print("=" * 80)
    print("Dataset:", args.dataset, "Backbone:", args.backbone, "Training:", args.training_mode)
    print("Baseline:", baseline, "Attack algo:", algo, "Epsilon:", epsilon, "Alpha:", alpha, "Steps:", args.steps)
    print("Target label:", args.target_label)
    print("Model:", args.model_name)
    print("TFLite:", tflite_path)
    print("SavedModel:", saved_model_path)
    print("Images:", image_dir)
    print("=" * 80)

    interpreter = load_interpreter(tflite_path)
    images_np, victim_labels_np, names = load_target_images(image_dir, interpreter, args.num_images, target_label=args.target_label)
    source_model = build_source_model(baseline, args.backbone, saved_model_path)
    fmodel = fb.TensorFlowModel(source_model, bounds=(0, 255))
    images_tf = tf.convert_to_tensor(images_np, dtype=tf.float32)

    if baseline == "PMA":
        source_labels_np = np.argmax(source_model(images_tf, training=False).numpy(), axis=1).astype(np.int64)
    else:
        source_labels_np = victim_labels_np.astype(np.int64)

    images = ep.astensor(images_tf)
    labels_tf = tf.convert_to_tensor(source_labels_np, dtype=tf.int64)
    labels = ep.astensor(labels_tf)
    attack, attack_name = build_attack(algo, steps=args.steps)

    success_rates = []
    for r in tqdm(range(args.rounds)):
        if algo in ["PGD", "MIFGSM"]:
            batch_size = max(1, int(args.attack_batch_size))
            adv_batches = []
            success_batches = []
            n_images = int(images_tf.shape[0])
            print("Attack mini-batch size:", batch_size, "Total images:", n_images)
            for start in range(0, n_images, batch_size):
                end = min(start + batch_size, n_images)
                x_b = images_tf[start:end]
                y_b = labels_tf[start:end]
                if algo == "PGD":
                    adv_b, success_b = pgd_attack_tf(source_model, x_b, y_b, epsilon, alpha, args.steps, random_start=args.random_start)
                else:
                    adv_b, success_b = mifgsm_attack_tf(source_model, x_b, y_b, epsilon, alpha, args.steps, decay=args.decay)
                adv_batches.append(adv_b)
                success_batches.append(tf.cast(success_b, tf.float32))
            adv_tf = tf.concat(adv_batches, axis=0)
            success_all = tf.concat(success_batches, axis=0)
            advs_list = [adv_tf]
            source_success = float(tf.reduce_mean(success_all).numpy())
        else:
            raw, advs_list, success = attack(fmodel, images, labels, epsilons=[epsilon])
            try:
                source_success = success.float32().mean().item()
            except Exception:
                source_success = success

        adv_dir, saved = save_advs(advs_list, args.dataset, args.backbone, args.training_mode, baseline, algo, args.model_name, r)
        adv_labels = infer_folder(interpreter, adv_dir)
        rate = calc_asr(victim_labels_np, adv_labels, target_label=args.target_label)
        success_rates.append(float(rate))
        print("Foolbox/source success:", source_success)
        print("TFLite victim attack success rate:", rate)

    result_dir = "results_json/{}/{}/{}/{}/{}".format(args.dataset, args.backbone, args.training_mode, baseline, algo)
    ensure(result_dir)
    result = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "training_mode": args.training_mode,
        "baseline": baseline,
        "attack_algo": algo,
        "attack_name": attack_name,
        "model_name": args.model_name,
        "epsilon": float(epsilon),
        "alpha": None if alpha is None else float(alpha),
        "steps": int(args.steps),
        "decay": float(args.decay),
        "target_label": int(args.target_label),
        "attack_batch_size": int(args.attack_batch_size),
        "rounds": args.rounds,
        "num_images": int(len(victim_labels_np)),
        "success_rates": success_rates,
        "mean_success_rate": float(np.mean(success_rates)),
        "min_success_rate": float(np.min(success_rates)),
        "max_success_rate": float(np.max(success_rates)),
    }
    out = "{}/{}_{}_{}_results.json".format(result_dir, baseline, algo, args.model_name)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("Saved result:", out)


if __name__ == "__main__":
    run_attack("BAMA")
