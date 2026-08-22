"""
Interfaz web mínima: el usuario sube una imagen y la app muestra si el
modelo predice "perro" o "gato" junto con el porcentaje de confianza.

Sin historial de sesión ni funcionalidades extra: alcance mínimo viable.
Ejecutar con: streamlit run app/streamlit_app.py

Importa `crear_modelo` del `model.py` local (ResNet18 en este proyecto), así
que la interfaz de usuario no necesita saber nada sobre la arquitectura
subyacente.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

DIRECTORIO_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DIRECTORIO_PROYECTO / "src"))

from dataset import NOMBRES_CLASES, crear_transformacion_evaluacion  # noqa: E402
from model import crear_modelo  # noqa: E402

RUTA_MEJOR_MODELO = DIRECTORIO_PROYECTO / "models" / "best_model.pth"
# Ambos archivos se generan localmente (`evaluate.py` y `train.py`) para que
# la app pueda mostrar métricas y curvas sin llamar al API de W&B en tiempo
# real: así nadie necesita credenciales de W&B solo para ver la app.
RUTA_METRICAS_TEST = DIRECTORIO_PROYECTO / "models" / "metricas_test.json"
RUTA_HISTORIAL_ENTRENAMIENTO = DIRECTORIO_PROYECTO / "models" / "historial_entrenamiento.json"


@st.cache_resource
def cargar_modelo() -> tuple[torch.nn.Module, torch.device]:
    """Carga el checkpoint entrenado una sola vez y lo reutiliza entre peticiones.

    `st.cache_resource` es el mecanismo recomendado por Streamlit para
    recursos no serializables como un modelo de PyTorch: evita recargarlo en
    cada interacción del usuario con la app.
    """
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = crear_modelo().to(dispositivo)
    estado_guardado = torch.load(RUTA_MEJOR_MODELO, map_location=dispositivo)
    modelo.load_state_dict(estado_guardado["state_dict_modelo"])
    modelo.eval()
    return modelo, dispositivo


def predecir_clase(imagen: Image.Image, modelo: torch.nn.Module, dispositivo: torch.device) -> tuple[str, float]:
    """Preprocesa la imagen igual que en entrenamiento y devuelve (clase, confianza)."""
    transformacion = crear_transformacion_evaluacion()
    tensor_imagen = transformacion(imagen.convert("RGB")).unsqueeze(0).to(dispositivo)

    with torch.no_grad():
        logit = modelo(tensor_imagen)
        probabilidad_perro = torch.sigmoid(logit).item()

    if probabilidad_perro >= 0.5:
        return NOMBRES_CLASES[1], probabilidad_perro
    return NOMBRES_CLASES[0], 1 - probabilidad_perro


def mostrar_metricas_y_curvas_entrenamiento() -> None:
    """Muestra las métricas finales de test y las curvas de entrenamiento.

    Lee dos archivos JSON "horneados" localmente (`metricas_test.json` por
    `evaluate.py` e `historial_entrenamiento.json` por `train.py`) en vez de
    consultar el API de W&B en tiempo real, para que la app funcione sin
    credenciales ni conexión a internet. Si algún archivo no existe todavía
    (por ejemplo, alguien clonó el repo sin haber corrido esos scripts), se
    informa con un mensaje simple en vez de que la app falle.
    """
    if not RUTA_METRICAS_TEST.exists() or not RUTA_HISTORIAL_ENTRENAMIENTO.exists():
        st.info(
            "Aún no hay datos de entrenamiento disponibles. Corre "
            "`python src/train.py` y luego `python src/evaluate.py` para generarlos."
        )
        return

    try:
        with open(RUTA_METRICAS_TEST, encoding="utf-8") as archivo:
            metricas_test = json.load(archivo)
        with open(RUTA_HISTORIAL_ENTRENAMIENTO, encoding="utf-8") as archivo:
            historial_entrenamiento = json.load(archivo)
    except (json.JSONDecodeError, OSError) as error:
        st.warning(f"No se pudieron leer los datos de entrenamiento guardados: {error}")
        return

    claves_metricas_esperadas = {"accuracy", "precision", "recall", "f1"}
    if not claves_metricas_esperadas.issubset(metricas_test) or not historial_entrenamiento:
        st.warning(
            "Los archivos de métricas/historial existen pero no tienen el "
            "formato esperado. Vuelve a correr `evaluate.py`/`train.py`."
        )
        return

    st.markdown("**Métricas finales sobre el test set**")
    columna_accuracy, columna_precision, columna_recall, columna_f1 = st.columns(4)
    columna_accuracy.metric("Accuracy", f"{metricas_test['accuracy'] * 100:.1f}%")
    columna_precision.metric("Precision", f"{metricas_test['precision'] * 100:.1f}%")
    columna_recall.metric("Recall", f"{metricas_test['recall'] * 100:.1f}%")
    columna_f1.metric("F1", f"{metricas_test['f1'] * 100:.1f}%")

    tabla_historial = pd.DataFrame(historial_entrenamiento).set_index("epoca")

    st.markdown("**Accuracy por época (entrenamiento vs. validación)**")
    st.line_chart(tabla_historial[["train_accuracy", "val_accuracy"]])

    st.markdown("**Pérdida por época (entrenamiento vs. validación)**")
    st.line_chart(tabla_historial[["train_loss", "val_loss"]])


def main() -> None:
    st.set_page_config(page_title="Clasificador perro vs. gato CNN", page_icon="🐾")
    st.title("Clasificador de perros y gatos — CNN")
    st.write("Sube una imagen de un perro o un gato y el modelo predice la clase.")

    with st.expander("Sobre el modelo"):
        st.markdown(
            "Este clasificador usa una **red neuronal convolucional (CNN)**: "
            "un tipo de modelo pensado para trabajar con imágenes, que "
            "aprende a reconocer patrones visuales (bordes, texturas, formas) "
            "aplicando pequeños filtros sobre la imagen en vez de mirar cada "
            "píxel por separado.\n\n"
            "En concreto, usa la arquitectura **ResNet18**, conocida por sus "
            "*conexiones residuales* (atajos que ayudan a la red a entrenarse "
            "mejor incluso siendo profunda). En este proyecto la red se "
            "entrenó **completamente desde cero**, sin usar pesos "
            "preentrenados: aprendió a distinguir perros de gatos únicamente "
            "a partir de las imágenes de este dataset."
        )

    with st.expander("Métricas y curvas de entrenamiento"):
        mostrar_metricas_y_curvas_entrenamiento()

    if not RUTA_MEJOR_MODELO.exists():
        st.error(
            f"No se encontró el modelo entrenado en '{RUTA_MEJOR_MODELO}'. "
            "Ejecuta `python src/train.py` antes de usar la app."
        )
        return

    archivo_subido = st.file_uploader("Imagen (JPG o PNG)", type=["jpg", "jpeg", "png"])
    if archivo_subido is None:
        return

    try:
        imagen = Image.open(archivo_subido)
    except UnidentifiedImageError:
        st.error("El archivo subido no es una imagen válida. Intenta con un JPG o PNG.")
        return

    st.image(imagen, caption="Imagen subida", use_container_width=True)

    modelo, dispositivo = cargar_modelo()
    clase_predicha, confianza = predecir_clase(imagen, modelo, dispositivo)

    st.subheader(f"Predicción: {clase_predicha}")
    st.metric("Confianza", f"{confianza * 100:.1f}%")


if __name__ == "__main__":
    main()
