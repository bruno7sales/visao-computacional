import argparse
import time
from pathlib import Path

import cv2

from image_classifier.settings import DATASET_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Coleta imagens pela webcam para montar o dataset.")
    parser.add_argument("--class-name", required=True, help="Nome da classe, por exemplo: caneca.")
    parser.add_argument("--count", type=int, default=50, help="Quantidade de imagens a salvar.")
    parser.add_argument("--camera", type=int, default=0, help="Indice da webcam.")
    parser.add_argument("--auto-interval", type=float, default=0.35, help="Intervalo da captura automatica.")
    return parser.parse_args()


def draw_overlay(frame, class_name, saved, total, auto_mode):
    status = "AUTO" if auto_mode else "MANUAL"
    lines = [
        f"Classe: {class_name}",
        f"Fotos: {saved}/{total}",
        f"Modo: {status}",
        "ESPACO salva | A auto | Q sair",
    ]
    for index, text in enumerate(lines):
        y = 30 + index * 28
        cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4)
        cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
    return frame


def save_frame(frame, output_dir: Path, class_name: str, image_number: int):
    timestamp = int(time.time() * 1000)
    filename = output_dir / f"{class_name}_{timestamp}_{image_number:03d}.jpg"
    cv2.imwrite(str(filename), frame)


def main():
    args = parse_args()
    output_dir = DATASET_DIR / args.class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Nao foi possivel abrir a webcam.")

    saved = len(list(output_dir.glob("*.jpg")))
    target = saved + args.count
    auto_mode = False
    last_capture = 0.0

    print("Janela aberta. Pressione ESPACO para salvar, A para automatico, Q para sair.")

    while saved < target:
        ok, frame = cap.read()
        if not ok:
            break

        preview = draw_overlay(frame.copy(), args.class_name, saved, target, auto_mode)
        cv2.imshow("Coleta de dataset", preview)

        key = cv2.waitKey(1) & 0xFF
        now = time.time()

        should_save = key == ord(" ") or (auto_mode and now - last_capture >= args.auto_interval)
        if should_save:
            saved += 1
            last_capture = now
            save_frame(frame, output_dir, args.class_name, saved)
            print(f"Imagem salva: {saved}/{target}")

        if key in (ord("a"), ord("A")):
            auto_mode = not auto_mode
            last_capture = 0.0

        if key in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Coleta finalizada. Total de imagens em {output_dir}: {len(list(output_dir.glob('*.jpg')))}")


if __name__ == "__main__":
    main()
