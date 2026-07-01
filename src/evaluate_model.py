import argparse

from image_classifier.datasets import load_eval_dataset
from image_classifier.inference import load_class_names, load_trained_model
from image_classifier.metrics import build_confusion_matrix
from image_classifier.plots import save_confusion_matrix
from image_classifier.settings import DATASET_DIR, MODEL_PATH


def parse_args():
    parser = argparse.ArgumentParser(description="Avalia o modelo treinado e gera matriz de confusao.")
    parser.add_argument("--dataset", default=str(DATASET_DIR), help="Pasta do dataset.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Arquivo .h5 do modelo.")
    return parser.parse_args()


def main():
    args = parse_args()
    class_names = load_class_names()
    dataset = load_eval_dataset(args.dataset)
    model = load_trained_model(args.model)
    loss, keras_accuracy = model.evaluate(dataset, verbose=0)
    matrix, accuracy = build_confusion_matrix(model, dataset, len(class_names))
    matrix_path = save_confusion_matrix(matrix, class_names)

    print("Avaliacao finalizada.")
    print(f"Acuracia: {accuracy:.2%}")
    print(f"Acuracia Keras: {keras_accuracy:.2%}")
    print(f"Perda: {loss:.4f}")
    print(f"Matriz de confusao salva em: {matrix_path}")


if __name__ == "__main__":
    main()
