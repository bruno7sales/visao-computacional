import argparse
import json

from image_classifier.datasets import load_train_validation_test_datasets
from image_classifier.model import build_model
from image_classifier.plots import save_training_plot
from image_classifier.settings import CLASSES_PATH, DATASET_DIR, MODEL_PATH, MODELS_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Treina uma CNN para classificar imagens do dataset.")
    parser.add_argument("--epochs", type=int, default=15, help="Quantidade de epocas de treinamento.")
    parser.add_argument("--dataset", default=str(DATASET_DIR), help="Pasta do dataset.")
    return parser.parse_args()


def main():
    args = parse_args()
    train_ds, validation_ds, test_ds, class_names = load_train_validation_test_datasets(args.dataset)

    if len(class_names) < 2:
        raise ValueError("O dataset precisa ter pelo menos 2 classes.")

    model = build_model(num_classes=len(class_names))
    model.summary()

    history = model.fit(train_ds, validation_data=validation_ds, epochs=args.epochs)
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    CLASSES_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    training_plot_path = save_training_plot(history)

    print("\nTreinamento finalizado.")
    print(f"Classes: {', '.join(class_names)}")
    print(f"Acuracia no teste: {test_accuracy:.2%}")
    print(f"Perda no teste: {test_loss:.4f}")
    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Grafico salvo em: {training_plot_path}")


if __name__ == "__main__":
    main()
