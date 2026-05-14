# -*- coding: utf-8 -*-
import os
import tensorflow as tf

def lite_converter(saved_model_dir, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    return output_path
