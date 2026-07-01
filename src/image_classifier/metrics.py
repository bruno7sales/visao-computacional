import numpy as np


def build_confusion_matrix(model, dataset, class_count):
    matrix = np.zeros((class_count, class_count), dtype=int)
    total = 0
    correct = 0

    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        labels_np = labels.numpy()
        for actual, predicted in zip(labels_np, predictions):
            matrix[actual, predicted] += 1
            total += 1
            correct += int(actual == predicted)

    accuracy = correct / total if total else 0.0
    return matrix, accuracy
