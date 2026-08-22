"""
Interfaz web mínima: el usuario sube una imagen y la app muestra si el
modelo predice "perro" o "gato" junto con el porcentaje de confianza.

Sin historial de sesión ni funcionalidades extra: alcance mínimo viable.
Ejecutar con: streamlit run app/streamlit_app.py

Este archivo es idéntico al de `clasificador-perros-gatos`: importa
`crear_modelo` del `model.py` local (ResNet18 en este proyecto), así que la
interfaz de usuario no necesita saber nada sobre la arquitectura subyacente.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

DIRECTORIO_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DIRECTORIO_PROYECTO / "src"))

from dataset import NOMBRES_CLASES, crear_transformacion_evaluacion  # noqa: E402
from model import crear_modelo  # noqa: E402

RUTA_MEJOR_MODELO = DIRECTORIO_PROYECTO / "models" / "best_model.pth"


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


def main() -> None:
    st.set_page_config(page_title="Clasificador perro vs. gato CNN", page_icon="🐾")
    st.title("Clasificador de perros y gatos — CNN")
    st.write("Sube una imagen de un perro o un gato y el modelo predice la clase.")

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
