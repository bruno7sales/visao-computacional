import argparse

import cv2

from image_classifier.inference import load_class_names, load_trained_model, predict_frame
from image_classifier.settings import MODEL_PATH


def parse_args():
    parser = argparse.ArgumentParser(description="Reconhecimento ao vivo usando webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Indice da webcam.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Arquivo .h5 do modelo.")
    return parser.parse_args()


def draw_prediction(frame, class_name, confidence):
    label = f"{class_name} ({confidence:.1%})"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 72), (20, 55, 84), -1)
    cv2.putText(frame, label, (18, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    cv2.putText(frame, "Pressione Q para sair", (18, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def main():
    args = parse_args()
    class_names = load_class_names()
    model = load_trained_model(args.model)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Nao foi possivel abrir a webcam.")

    print("Reconhecimento iniciado. Pressione Q para sair.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        class_name, confidence = predict_frame(model, frame, class_names)
        frame = draw_prediction(frame, class_name, confidence)
        cv2.imshow("Reconhecimento ao vivo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
