import time
from pathlib import Path

import cv2

try:
    cv2.setLogLevel(0)
except AttributeError:
    pass


def open_camera(camera_index):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(camera_index, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(camera_index)


def list_available_cameras(max_index=5):
    cameras = []
    for index in range(max_index + 1):
        cap = open_camera(index)
        ok, _ = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok:
            cameras.append(index)
    return cameras


def capture_frame(camera_index):
    cap = open_camera(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a camera {camera_index}.")

    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Nao foi possivel capturar imagem da camera {camera_index}.")
    return frame


def save_frame(frame, output_dir: Path, class_name: str, image_number: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    filename = output_dir / f"{class_name}_{timestamp}_{image_number:03d}.jpg"
    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError(f"Falha ao codificar a imagem para JPG antes de salvar em {filename}.")

    with open(filename, "wb") as f:
        f.write(encoded.tobytes())

    if not filename.exists() or filename.stat().st_size == 0:
        raise RuntimeError(f"Falha ao salvar a imagem em {filename}.")
    return filename


def frame_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
