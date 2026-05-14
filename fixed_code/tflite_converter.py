# -*- coding: utf-8 -*-
import os
import tensorflow as tf

def lite_converter(saved_model_dir, output_path):
    """Convert TensorFlow SavedModel to a float32 TFLite model."""
    if not os.path.exists(saved_model_dir):
        raise FileNotFoundError("SavedModel directory not found: {}".format(saved_model_dir))
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = []
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print("Saved TFLite:", output_path)
