import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

if get_script_run_ctx(suppress_warning=True) is None:
    print("Iniciando a interface com Streamlit...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    sys.exit(0)

from image_classifier.camera import frame_to_rgb, save_frame
from image_classifier.inference import load_class_names, load_trained_model, predict_frame
from image_classifier.settings import CLASSES_PATH, DATASET_DIR, MODEL_PATH, RESULTS_DIR

# Optional: streamlit-webrtc for browser-based live video processing
try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
except Exception:
    webrtc_streamer = None
    VideoTransformerBase = None


st.set_page_config(page_title="Classificador CNN", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.25rem; max-width: 1180px; }
    div[data-testid="stMetric"] { border: 1px solid #dde5ec; padding: 0.75rem; border-radius: 8px; }
    .app-panel { border: 1px solid #d8e2ea; border-radius: 8px; padding: 0.85rem; background: #f7fafc; }
    .small-note { color: #52606d; font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def cached_model(model_updated_at):
    return load_trained_model(), load_class_names()


def run_script(script_name, *args):
    command = [sys.executable, str(DATASET_DIR.parent / "src" / script_name), *map(str, args)]
    return subprocess.run(command, cwd=DATASET_DIR.parent, capture_output=True, text=True)


def count_images(class_name):
    class_dir = DATASET_DIR / class_name
    if not class_dir.exists():
        return 0
    return len(list(class_dir.glob("*.jpg")))


def dataset_classes():
    if not DATASET_DIR.exists():
        return []
    return sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir())


def dataset_total_images():
    return sum(count_images(class_name) for class_name in dataset_classes())


def normalize_class_name(value):
    return value.strip().lower().replace(" ", "_")


def model_is_ready():
    return MODEL_PATH.exists() and CLASSES_PATH.exists()


def get_model_updated_at():
    return MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0


def get_model_for_app():
    if not model_is_ready():
        return None, None
    return cached_model(get_model_updated_at())


def draw_prediction(frame, class_name, confidence):
    label = f"{class_name} ({confidence:.1%})"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 72), (20, 55, 84), -1)
    cv2.putText(frame, label, (18, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return frame


def camera_file_to_frame(camera_file):
    data = np.frombuffer(camera_file.getvalue(), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Nao foi possivel ler a imagem enviada pela webcam.")
    return frame


def camera_buffer_to_frame(buffer: bytes):
    data = np.frombuffer(buffer, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Nao foi possivel ler os bytes da imagem.")
    return frame


def save_buffer_to_jpg(buffer: bytes, output_dir: Path, class_name: str, image_number: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    filename = output_dir / f"{class_name}_{timestamp}_{image_number:03d}.jpg"
    with open(filename, "wb") as f:
        f.write(buffer)
    if not filename.exists() or filename.stat().st_size == 0:
        raise RuntimeError(f"Falha ao salvar a imagem em {filename}.")
    return filename


if "preview_frame" not in st.session_state:
    st.session_state.preview_frame = None
if "preview_buffer" not in st.session_state:
    st.session_state.preview_buffer = None
if "preview_camera" not in st.session_state:
    st.session_state.preview_camera = None


with st.sidebar:
    st.title("Menu")
    page = st.radio(
        "Navegacao",
        ["Visao geral", "Captura e dataset", "Treino", "Avaliacao", "Reconhecimento"],
        label_visibility="collapsed",
    )

    st.divider()
    st.header("Webcam")
    camera_source = "Webcam do navegador"
    st.info("A webcam sera solicitada pelo navegador nas telas de captura/reconhecimento.")

    st.divider()
    st.caption(f"Modelo: {MODEL_PATH.name if MODEL_PATH.exists() else 'nao treinado'}")
    st.caption(f"Classes: {', '.join(dataset_classes()) or 'nenhuma'}")


st.title("Classificador de imagens com CNN")

if page == "Visao geral":
    st.subheader("Painel do projeto")
    c1, c2, c3 = st.columns(3)
    c1.metric("Classes", len(dataset_classes()))
    c2.metric("Imagens", dataset_total_images())
    c3.metric("Modelo", "pronto" if model_is_ready() else "pendente")

    st.markdown(
        """
        <div class="app-panel">
        O fluxo completo esta no menu lateral: escolha a camera, capture imagens para cada classe,
        treine a CNN, avalie o resultado e use o reconhecimento ao vivo.
        </div>
        """,
        unsafe_allow_html=True,
    )

    classes = dataset_classes()
    if classes:
        st.subheader("Dataset atual")
        for class_name in classes:
            st.progress(min(count_images(class_name) / 50, 1.0), text=f"{class_name}: {count_images(class_name)} imagens")
    else:
        st.info("Crie pelo menos duas classes em Captura e dataset para liberar o treino.")

elif page == "Captura e dataset":
    st.subheader("Captura e dataset")
    left, right = st.columns([0.9, 1.1])

    with right:
        frame_slot = st.empty()
        status_slot = st.empty()

        webcam_file = st.camera_input("Webcam", help="Permita o uso da webcam no navegador e tire uma foto para usar no dataset.")
        if webcam_file is not None:
            try:
                buffer = webcam_file.getvalue()
                frame = camera_buffer_to_frame(buffer)
                st.session_state.preview_buffer = buffer
                st.session_state.preview_frame = frame
                st.session_state.preview_camera = "webcam do navegador"
                frame_slot.image(frame_to_rgb(frame), channels="RGB", use_container_width=True, caption="Foto capturada pela webcam")
                status_slot.info("Imagem capturada. Use o botão de salvar para gravar no dataset.")
            except RuntimeError as exc:
                status_slot.error(str(exc))
        elif st.session_state.preview_frame is not None:
            frame_slot.image(frame_to_rgb(st.session_state.preview_frame), channels="RGB", use_container_width=True, caption="Previa atual da webcam")
            status_slot.info("Clique em 'Salvar foto da previa' para armazenar a imagem capturada no dataset.")
        else:
            status_slot.info("Use a webcam no navegador para capturar uma imagem e salvar no dataset.")

    with left:
        class_name = normalize_class_name(st.text_input("Nome da classe", placeholder="ex.: caneca"))
        save_preview = st.button(
            "Salvar foto da previa",
            use_container_width=True,
            disabled=not class_name or st.session_state.preview_buffer is None,
        )

        if class_name:
            st.metric("Fotos da classe", count_images(class_name))

        if save_preview and class_name and st.session_state.preview_buffer is not None:
            try:
                image_number = count_images(class_name) + 1
                output_path = save_buffer_to_jpg(st.session_state.preview_buffer, DATASET_DIR / class_name, class_name, image_number)
                saved_size = output_path.stat().st_size if output_path.exists() else 0
                status_slot.success(f"Imagem salva em {output_path} ({saved_size} bytes).")
                st.session_state.preview_frame = None
                st.session_state.preview_buffer = None
                st.rerun()
            except Exception as exc:
                status_slot.error(f"Falha ao salvar a imagem: {exc}")

elif page == "Treino":
    st.subheader("Treino da CNN")
    c1, c2, c3 = st.columns(3)
    c1.metric("Classes", len(dataset_classes()))
    c2.metric("Imagens", dataset_total_images())
    c3.metric("Modelo", "pronto" if model_is_ready() else "pendente")

    epochs = st.number_input("Epocas", min_value=1, max_value=100, value=15, step=1)
    train_button = st.button("Treinar modelo", use_container_width=True, disabled=len(dataset_classes()) < 2)

    if len(dataset_classes()) < 2:
        st.info("Cadastre pelo menos duas classes no dataset antes de treinar.")

    if train_button:
        with st.spinner("Treinando modelo..."):
            result = run_script("train_cnn.py", "--epochs", int(epochs))
        if result.returncode == 0:
            cached_model.clear()
            st.success("Treinamento finalizado.")
            st.code(result.stdout)
            training_plot = RESULTS_DIR / "treinamento.png"
            if training_plot.exists():
                st.image(str(training_plot), use_container_width=True)
        else:
            st.error("Falha no treinamento.")
            st.code(result.stderr or result.stdout)

elif page == "Avaliacao":
    st.subheader("Avaliacao do modelo")
    evaluate_button = st.button("Avaliar modelo", use_container_width=True, disabled=not model_is_ready())

    if not model_is_ready():
        st.info("Treine o modelo antes de avaliar.")

    if evaluate_button:
        with st.spinner("Avaliando modelo..."):
            result = run_script("evaluate_model.py")
        if result.returncode == 0:
            st.success("Avaliacao finalizada.")
            st.code(result.stdout)
            confusion_matrix = RESULTS_DIR / "matriz_confusao.png"
            if confusion_matrix.exists():
                st.image(str(confusion_matrix), use_container_width=True)
        else:
            st.error("Falha na avaliacao.")
            st.code(result.stderr or result.stdout)

elif page == "Reconhecimento":
    st.subheader("Reconhecimento pela camera")
    model, class_names = get_model_for_app()
    left, right = st.columns([0.8, 1.2])

    with left:
        if model is None:
            st.info("Treine o modelo antes de iniciar o reconhecimento.")
        else:
            st.markdown("Escolha entre reconhecimento por foto (captura) ou em tempo real (Ao Vivo).")

    with right:
        live_mode = st.checkbox("Reconhecimento em tempo real (Ao Vivo)")

        frame_slot = st.empty()
        metric_col, confidence_col = st.columns(2)

        if live_mode:
            if VideoTransformerBase is None or webrtc_streamer is None:
                st.error("Dependência `streamlit-webrtc` não encontrada. Instale com `pip install streamlit-webrtc` e reinicie a app.")
            elif model is None:
                st.info("Treine o modelo antes de iniciar o reconhecimento ao vivo.")
            else:
                import av

                class LiveTransformer(VideoTransformerBase):
                    def __init__(self):
                        self.model = model
                        self.class_names = class_names

                    def recv(self, frame):
                        img = frame.to_ndarray(format="bgr24")
                        try:
                            class_name, confidence = predict_frame(self.model, img, self.class_names)
                            img = draw_prediction(img, class_name, confidence)
                        except Exception:
                            pass
                        return av.VideoFrame.from_ndarray(img, format="bgr24")

                webrtc_streamer(key="reconhecimento_ao_vivo", video_transformer_factory=LiveTransformer)

        else:
            webcam_file = st.camera_input("Webcam para reconhecimento", help="Tire uma foto pela webcam do navegador para classificar.")
            if webcam_file is not None:
                try:
                    frame = camera_file_to_frame(webcam_file)
                    if model is not None:
                        class_name, confidence = predict_frame(model, frame, class_names)
                        frame = draw_prediction(frame, class_name, confidence)
                        metric_col.metric("Classe", class_name)
                        confidence_col.metric("Confianca", f"{confidence:.1%}")
                    frame_slot.image(frame_to_rgb(frame), channels="RGB", use_container_width=True, caption="Imagem da webcam do navegador")
                except RuntimeError as exc:
                    st.error(str(exc))
