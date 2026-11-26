"""
App Streamlit: clasificador Hombre / Mujer por cámara
Modelo propio YOLO11-cls (entrenado en Colab) + DeepFace opcional (ensemble).
"""
import threading
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ---------------- CONFIGURACIÓN ----------------
RUTA_MODELO = Path(__file__).parent / "modelo_genero_yolo.pt"
IMG_SIZE = 224
COLOR_HOMBRE = (255, 128, 0)   # BGR (azul)
COLOR_MUJER = (203, 0, 255)    # BGR (rosa)
# ------------------------------------------------

st.set_page_config(page_title="Clasificador de género", page_icon="🎥", layout="centered")


# ================== CARGA DE MODELOS (una sola vez) ==================
@st.cache_resource(show_spinner="Cargando modelo YOLO...")
def cargar_yolo():
    from ultralytics import YOLO
    return YOLO(str(RUTA_MODELO))


@st.cache_resource(show_spinner="Cargando DeepFace (la primera vez descarga pesos)...")
def cargar_deepface():
    try:
        from deepface import DeepFace
        # Calentamiento: descarga y deja en memoria el modelo de género
        DeepFace.analyze(np.zeros((224, 224, 3), dtype=np.uint8), actions=["gender"],
                         detector_backend="skip", enforce_detection=False, silent=True)
        return DeepFace
    except Exception as e:
        st.warning(f"DeepFace no disponible, se usará solo YOLO. ({e})")
        return None


@st.cache_resource
def cargar_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ================== LÓGICA DE PREDICCIÓN ==================
LOCK = threading.Lock()  # el video en vivo corre en otro hilo


def detectar_rostros(img_bgr, detector):
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return detector.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))


def recortar(img, x, y, w, h, margen=0.25):
    H, W = img.shape[:2]
    mx, my = int(w * margen), int(h * margen)
    return img[max(0, y - my):min(H, y + h + my), max(0, x - mx):min(W, x + w + mx)]


def pred_yolo(rostro_bgr, yolo):
    r = yolo(rostro_bgr, imgsz=IMG_SIZE, verbose=False)[0]
    probs = r.probs.data.cpu().numpy()
    d = {r.names[i]: float(p) * 100 for i, p in enumerate(probs)}
    return {"hombre": d.get("hombre", 0.0), "mujer": d.get("mujer", 0.0)}


def pred_deepface(rostro_bgr, DeepFace):
    res = DeepFace.analyze(rostro_bgr, actions=["gender"], detector_backend="skip",
                           enforce_detection=False, silent=True)
    g = res[0]["gender"]
    return {"hombre": float(g["Man"]), "mujer": float(g["Woman"])}


def analizar(img_bgr, yolo, detector, DeepFace=None, cache_df=None, usar_df_ahora=True):
    """Devuelve lista de dicts con bbox, etiqueta y porcentaje predominante."""
    resultados = []
    for i, (x, y, w, h) in enumerate(detectar_rostros(img_bgr, detector)):
        rostro = recortar(img_bgr, x, y, w, h)
        with LOCK:
            final = pred_yolo(rostro, yolo)
            if DeepFace is not None:
                pdf = None
                if usar_df_ahora or cache_df is None or i not in cache_df:
                    try:
                        pdf = pred_deepface(rostro, DeepFace)
                        if cache_df is not None:
                            cache_df[i] = pdf
                    except Exception:
                        pass
                elif cache_df is not None:
                    pdf = cache_df.get(i)
                if pdf:
                    final = {k: (final[k] + pdf[k]) / 2 for k in final}

        hm, mj = final["hombre"], final["mujer"]
        etiqueta, pct = ("Hombre", hm) if hm >= mj else ("Mujer", mj)
        resultados.append({"bbox": (x, y, w, h), "etiqueta": etiqueta, "pct": pct})
    return resultados


def dibujar(img_bgr, resultados):
    img = img_bgr.copy()
    for r in resultados:
        x, y, w, h = r["bbox"]
        color = COLOR_HOMBRE if r["etiqueta"] == "Hombre" else COLOR_MUJER
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        cv2.rectangle(img, (x, max(0, y - 30)), (x + w, y), color, -1)
        cv2.putText(img, f"{r['etiqueta']}: {r['pct']:.0f}%", (x + 4, max(20, y - 9)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def mostrar_resultado(img_bgr, yolo, detector, DeepFace):
    res = analizar(img_bgr, yolo, detector, DeepFace)
    st.image(cv2.cvtColor(dibujar(img_bgr, res), cv2.COLOR_BGR2RGB), use_container_width=True)
    if not res:
        st.warning("😕 No se detectó ningún rostro. Acércate y mejora la iluminación.")
    for i, r in enumerate(res, 1):
        st.metric(f"Rostro {i}", f"{r['etiqueta']}", f"{r['pct']:.1f}%", delta_color="off")


def pil_a_bgr(archivo):
    return cv2.cvtColor(np.array(Image.open(archivo).convert("RGB")), cv2.COLOR_RGB2BGR)


# ================== INTERFAZ ==================
st.title("🎥 Clasificador de género")
st.caption("Modelo YOLO11-cls entrenado con FairFace + DeepFace. Muestra la clase predominante.")

if not RUTA_MODELO.exists():
    st.error(f"No se encontró el modelo `{RUTA_MODELO.name}`. Colócalo junto a app.py.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Opciones")
    usar_deepface = st.toggle("Usar DeepFace (ensemble)", value=True,
                              help="Más preciso, pero más lento y usa más memoria.")
    df_cada_n = st.slider("En vivo: DeepFace cada N cuadros", 1, 15, 5)
    st.info("Estima el sexo aparente según rasgos faciales (binario). "
            "No mide identidad de género. Demo educativa.")

yolo = cargar_yolo()
detector = cargar_detector()
DeepFace = cargar_deepface() if usar_deepface else None

tab_foto, tab_vivo, tab_subir = st.tabs(["📸 Foto", "🎥 En vivo", "🖼️ Subir imagen"])

# ---------- Foto con la cámara ----------
with tab_foto:
    foto = st.camera_input("Toma una foto")
    if foto is not None:
        mostrar_resultado(pil_a_bgr(foto), yolo, detector, DeepFace)

# ---------- Video en vivo (WebRTC) ----------
with tab_vivo:
    try:
        import av
        from streamlit_webrtc import webrtc_streamer, WebRtcMode

        estado = {"frame": 0, "cache_df": {}}

        def callback(frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            estado["frame"] += 1
            usar_ahora = estado["frame"] % df_cada_n == 0
            res = analizar(img, yolo, detector, DeepFace,
                           cache_df=estado["cache_df"], usar_df_ahora=usar_ahora)
            return av.VideoFrame.from_ndarray(dibujar(img, res), format="bgr24")

        webrtc_streamer(
            key="genero-en-vivo",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=callback,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )
        st.caption("Presiona **START** y permite el acceso a la cámara. "
                   "Si no conecta en tu red, usa la pestaña 📸 Foto.")
    except ImportError:
        st.warning("Instala `streamlit-webrtc` y `av` para el modo en vivo.")

# ---------- Subir imagen ----------
with tab_subir:
    archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png", "webp"])
    if archivo is not None:
        mostrar_resultado(pil_a_bgr(archivo), yolo, detector, DeepFace)
