import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("ABSL_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import tensorflow as tf

from image_classifier.settings import BATCH_SIZE, IMAGE_SIZE, RANDOM_SEED


def _list_image_paths(dataset_dir):
    dataset_dir = Path(dataset_dir)
    class_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    class_names = [p.name for p in class_dirs]

    image_paths = []
    image_labels = []
    for label, class_dir in enumerate(class_dirs):
        for extension in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"):
            for image_path in class_dir.glob(extension):
                image_paths.append(str(image_path))
                image_labels.append(label)

    if not image_paths:
        raise ValueError(f"Nenhuma imagem encontrada em {dataset_dir}")

    order = np.arange(len(image_paths))
    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(order)
    image_paths = [image_paths[i] for i in order]
    image_labels = [image_labels[i] for i in order]

    return image_paths, image_labels, class_names


def _parse_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def load_train_validation_test_datasets(dataset_dir):
    image_paths, image_labels, class_names = _list_image_paths(dataset_dir)
    total = len(image_paths)
    if total < 2:
        raise ValueError("O dataset precisa ter pelo menos 2 imagens.")

    validation_and_test_size = max(1, int(total * 0.3))
    train_size = total - validation_and_test_size

    train_paths = image_paths[:train_size]
    train_labels = image_labels[:train_size]
    validation_test_paths = image_paths[train_size:]
    validation_test_labels = image_labels[train_size:]

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    validation_and_test_ds = tf.data.Dataset.from_tensor_slices((validation_test_paths, validation_test_labels))

    train_ds = train_ds.map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    validation_and_test_ds = validation_and_test_ds.map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.batch(BATCH_SIZE)
    validation_and_test_ds = validation_and_test_ds.batch(BATCH_SIZE)

    validation_batches = tf.data.experimental.cardinality(validation_and_test_ds).numpy()
    if validation_batches <= 1:
        validation_ds = validation_and_test_ds
        test_ds = validation_and_test_ds
    else:
        test_batches = max(1, validation_batches // 2)
        test_ds = validation_and_test_ds.take(test_batches)
        validation_ds = validation_and_test_ds.skip(test_batches)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=RANDOM_SEED).prefetch(buffer_size=autotune)
    validation_ds = validation_ds.cache().prefetch(buffer_size=autotune)
    test_ds = test_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, validation_ds, test_ds, class_names


def load_eval_dataset(dataset_dir):
    image_paths, image_labels, _ = _list_image_paths(dataset_dir)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))
    dataset = dataset.map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset.prefetch(tf.data.AUTOTUNE)

