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

import altair as alt
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
RUTA_MATRIZ_CONFUSION = DIRECTORIO_PROYECTO / "models" / "matriz_confusion.png"


@st.cache_resource
def cargar_modelo() -> tuple[torch.nn.Module, torch.device]:
    """Carga el checkpoint entrenado una sola vez y lo reutiliza entre peticiones.

    `st.cache_resource` es el mecanismo recomendado por Streamlit para
    recursos no serializables como un modelo de PyTorch: evita recargarlo en
    cada interacción del usuario con la app.
    """
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = crear_modelo().to(dispositivo)
    # `weights_only=True` restringe la deserialización a tensores y tipos
    # simples (sin ejecutar pickle arbitrario), la práctica de seguridad
    # recomendada por PyTorch. El checkpoint que genera `train.py` solo
    # contiene tensores y tipos nativos (`state_dict_modelo`, `epoca_global`,
    # `val_loss`, `val_accuracy`, `fase`), así que no requiere
    # `add_safe_globals` para tipos custom.
    estado_guardado = torch.load(RUTA_MEJOR_MODELO, map_location=dispositivo, weights_only=True)
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


def construir_grafico_curvas_entrenamiento(
    tabla_historial: pd.DataFrame,
    columnas: tuple[str, str],
    nombres_series: tuple[str, str],
    titulo_eje_y: str,
) -> alt.Chart:
    """Construye un gráfico de líneas (Altair) con el dominio de los ejes fijado a mano.

    `st.line_chart` delega en Vega-Lite el autoescalado del dominio de los
    ejes según el tamaño del contenedor en el DOM. Cuando ese contenedor está
    oculto (por ejemplo, dentro de un `st.expander` colapsado, como ocurre en
    esta app), Vega-Lite no puede medirlo y calcula un rango inválido
    ("Infinity, -Infinity"), lo que genera warnings en la consola del
    navegador en cada carga de página aunque el usuario nunca haya abierto el
    expander. Al fijar el dominio de los ejes X e Y explícitamente (a partir
    de los propios datos), Vega-Lite ya no necesita medir el contenedor para
    autoescalar, y el warning desaparece.
    """
    columna_train, columna_val = columnas
    nombre_serie_train, nombre_serie_val = nombres_series

    # Formato largo (una fila por punto y serie) porque Altair codifica el
    # color de cada línea a partir de una columna categórica, no de columnas
    # separadas como espera `st.line_chart`.
    datos_formato_largo = tabla_historial.reset_index().melt(
        id_vars="epoca",
        value_vars=[columna_train, columna_val],
        var_name="serie",
        value_name="valor",
    )
    datos_formato_largo["serie"] = datos_formato_largo["serie"].map(
        {columna_train: nombre_serie_train, columna_val: nombre_serie_val}
    )

    epoca_minima = int(tabla_historial.index.min())
    epoca_maxima = int(tabla_historial.index.max())
    valor_minimo = float(tabla_historial[[columna_train, columna_val]].min().min())
    valor_maximo = float(tabla_historial[[columna_train, columna_val]].max().max())
    # Margen del 5% del rango de valores para que las curvas no queden
    # pegadas al borde superior/inferior del gráfico; si el rango es plano
    # (valor_maximo == valor_minimo), se usa un margen fijo pequeño.
    margen_eje_y = (valor_maximo - valor_minimo) * 0.05 or 0.05

    return (
        alt.Chart(datos_formato_largo)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "epoca:Q",
                title="Época",
                scale=alt.Scale(domain=[epoca_minima, epoca_maxima]),
            ),
            y=alt.Y(
                "valor:Q",
                title=titulo_eje_y,
                scale=alt.Scale(domain=[valor_minimo - margen_eje_y, valor_maximo + margen_eje_y]),
            ),
            color=alt.Color("serie:N", title=""),
        )
    )


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
    grafico_accuracy = construir_grafico_curvas_entrenamiento(
        tabla_historial,
        ("train_accuracy", "val_accuracy"),
        ("entrenamiento", "validación"),
        "Accuracy",
    )
    st.altair_chart(grafico_accuracy, width="stretch")

    st.markdown("**Pérdida por época (entrenamiento vs. validación)**")
    grafico_perdida = construir_grafico_curvas_entrenamiento(
        tabla_historial,
        ("train_loss", "val_loss"),
        ("entrenamiento", "validación"),
        "Pérdida",
    )
    st.altair_chart(grafico_perdida, width="stretch")

    st.markdown("**Matriz de confusión**")
    if RUTA_MATRIZ_CONFUSION.exists():
        st.image(str(RUTA_MATRIZ_CONFUSION), caption="Matriz de confusión sobre el test set", width="stretch")
    else:
        st.info(
            "Aún no hay matriz de confusión disponible. Corre "
            "`python src/evaluate.py` para generarla."
        )


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

    st.image(imagen, caption="Imagen subida", width="stretch")

    modelo, dispositivo = cargar_modelo()
    clase_predicha, confianza = predecir_clase(imagen, modelo, dispositivo)

    st.subheader(f"Predicción: {clase_predicha}")
    st.metric("Confianza", f"{confianza * 100:.1f}%")


if __name__ == "__main__":
    main()
