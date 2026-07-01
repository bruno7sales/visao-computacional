import json
import os

import cv2
import numpy as np

from image_classifier.settings import CLASSES_PATH, IMAGE_SIZE, MODEL_PATH


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def load_class_names(classes_path=CLASSES_PATH):
    if not classes_path.exists():
        raise FileNotFoundError("Arquivo classes.json nao encontrado. Treine o modelo primeiro.")
    return json.loads(classes_path.read_text(encoding="utf-8"))


def load_trained_model(model_path=MODEL_PATH):
    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


def predict_frame(model, frame, class_names):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMAGE_SIZE)
    batch = np.expand_dims(resized, axis=0)
    probabilities = model.predict(batch, verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    confidence = float(probabilities[class_index])
    return class_names[class_index], confidence
