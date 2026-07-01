import matplotlib.pyplot as plt
import numpy as np

from image_classifier.settings import RESULTS_DIR


def save_training_plot(history):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    accuracy = history.history["accuracy"]
    val_accuracy = history.history.get("val_accuracy", [])
    loss = history.history["loss"]
    val_loss = history.history.get("val_loss", [])
    epochs = range(1, len(accuracy) + 1)

    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, label="Treino")
    if val_accuracy:
        plt.plot(epochs, val_accuracy, label="Validacao")
    plt.title("Acuracia")
    plt.xlabel("Epoca")
    plt.ylabel("Acuracia")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Treino")
    if val_loss:
        plt.plot(epochs, val_loss, label="Validacao")
    plt.title("Perda")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    output_path = RESULTS_DIR / "treinamento.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def save_confusion_matrix(matrix, class_names):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Matriz de confusao")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Previsto")
    plt.ylabel("Real")

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            plt.text(col, row, matrix[row, col], ha="center", va="center", color=color)

    plt.tight_layout()
    output_path = RESULTS_DIR / "matriz_confusao.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
