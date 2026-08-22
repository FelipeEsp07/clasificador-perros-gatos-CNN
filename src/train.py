"""
Entrenamiento del clasificador perro/gato con ResNet18 **desde cero**, sin
pesos preentrenados de ImageNet (sin transfer learning).

Al no haber backbone preentrenado que proteger, no hay esquema de dos fases
(cabeza congelada → fine-tuning): todas las capas se entrenan juntas desde
la época 1, con un único optimizador. Se compensa con:
- Un learning rate inicial relativamente alto (1e-3), típico de entrenar
  una red convolucional desde cero.
- Muchas más épocas máximas (hasta 30), porque converger desde pesos
  aleatorios es sustancialmente más lento que ajustar un modelo ya
  preentrenado.
- Early stopping con paciencia algo mayor (6 épocas) para no cortar el
  entrenamiento prematuramente durante ese proceso de convergencia más lento.

El resto del pipeline (dataset, augmentation, criterio de pérdida,
optimizador AdamW, scheduler, checkpointing del mejor modelo por `val_loss`,
semillas) sigue el patrón estándar para este tipo de entrenamiento.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import crear_dataloaders
from model import contar_parametros_entrenables, crear_modelo

SEMILLA = 42
DIRECTORIO_PROYECTO = Path(__file__).resolve().parent.parent
DIRECTORIO_DATOS = DIRECTORIO_PROYECTO / "data"
RUTA_MEJOR_MODELO = DIRECTORIO_PROYECTO / "models" / "best_model.pth"
# La app de Streamlit lee este archivo local para graficar las curvas de
# entrenamiento sin depender del API de W&B (ver `guardar_historial_entrenamiento`).
RUTA_HISTORIAL_ENTRENAMIENTO = DIRECTORIO_PROYECTO / "models" / "historial_entrenamiento.json"

# Nombre del proyecto de W&B donde se registran las corridas de
# entrenamiento; los campos `backbone` y `pretrained` del config permiten
# filtrar/agrupar los runs desde el dashboard.
NOMBRE_PROYECTO_WANDB = "clasificador-perros-gatos"

CONFIGURACION = {
    "arquitectura": "resnet18",
    "backbone": "resnet18",
    # Campo explícito para identificar en W&B que este run entrena desde
    # cero (pesos aleatorios), sin transfer learning.
    "pretrained": False,
    "entrena_desde_cero": True,
    "tamano_lote": 32,
    "proporcion_validacion": 0.15,
    "tasa_aprendizaje": 1e-3,
    "weight_decay": 1e-4,
    "probabilidad_dropout": 0.3,
    "epocas_maximas": 30,
    "paciencia_early_stopping": 6,
    "semilla": SEMILLA,
}


def fijar_semillas(semilla: int) -> None:
    """Fija las semillas de los generadores aleatorios usados en el pipeline.

    Necesario para que el split train/val, la inicialización aleatoria de
    TODOS los pesos del modelo (no solo la cabeza, a diferencia de transfer
    learning) y el orden de mezclado de los lotes sean reproducibles entre
    corridas.
    """
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    torch.cuda.manual_seed_all(semilla)


def entrenar_una_epoca(
    modelo: nn.Module,
    cargador: DataLoader,
    criterio: nn.Module,
    optimizador: torch.optim.Optimizer,
    dispositivo: torch.device,
) -> tuple[float, float]:
    """Ejecuta una época de entrenamiento y devuelve (loss promedio, accuracy)."""
    modelo.train()  # activa Dropout y usa estadísticas de lote en BatchNorm
    perdida_acumulada = 0.0
    aciertos = 0
    total_ejemplos = 0

    for imagenes, etiquetas in tqdm(cargador, desc="entrenamiento", leave=False):
        imagenes = imagenes.to(dispositivo, non_blocking=True)
        etiquetas = etiquetas.to(dispositivo, non_blocking=True).unsqueeze(1)

        optimizador.zero_grad()
        logits = modelo(imagenes)
        perdida = criterio(logits, etiquetas)
        perdida.backward()
        optimizador.step()

        perdida_acumulada += perdida.item() * imagenes.size(0)
        predicciones = (torch.sigmoid(logits) >= 0.5).float()
        aciertos += (predicciones == etiquetas).sum().item()
        total_ejemplos += imagenes.size(0)

    return perdida_acumulada / total_ejemplos, aciertos / total_ejemplos


@torch.no_grad()
def evaluar(
    modelo: nn.Module,
    cargador: DataLoader,
    criterio: nn.Module,
    dispositivo: torch.device,
) -> tuple[float, float]:
    """Evalúa el modelo sin actualizar pesos y devuelve (loss promedio, accuracy)."""
    modelo.eval()  # desactiva Dropout y usa estadísticas globales en BatchNorm
    perdida_acumulada = 0.0
    aciertos = 0
    total_ejemplos = 0

    for imagenes, etiquetas in tqdm(cargador, desc="validación", leave=False):
        imagenes = imagenes.to(dispositivo, non_blocking=True)
        etiquetas = etiquetas.to(dispositivo, non_blocking=True).unsqueeze(1)

        logits = modelo(imagenes)
        perdida = criterio(logits, etiquetas)

        perdida_acumulada += perdida.item() * imagenes.size(0)
        predicciones = (torch.sigmoid(logits) >= 0.5).float()
        aciertos += (predicciones == etiquetas).sum().item()
        total_ejemplos += imagenes.size(0)

    return perdida_acumulada / total_ejemplos, aciertos / total_ejemplos


class _EjecucionSinWandb:
    """Sustituto sin-operación de `wandb.Run` para cuando el paquete no está instalado.

    Quien clona el repo sin cuenta de W&B puede entrenar igual: basta con no
    instalar `wandb` (o correr con `WANDB_MODE=disabled` si sí lo instaló) y
    el entrenamiento sigue funcionando idéntico, solo sin registrar métricas.
    """

    def __init__(self) -> None:
        self.summary: dict = {}

    def log(self, *_args, **_kwargs) -> None:
        return None

    def finish(self) -> None:
        return None


def registrar_metricas_en_wandb(ejecucion, metricas: dict) -> None:
    """Envía métricas a W&B tolerando fallos de conectividad del servicio local.

    En la máquina de desarrollo, el proceso local `wandb-core` (que sincroniza
    los datos con el servidor) ocasionalmente cierra la conexión de socket
    local con un `ConnectionAbortedError` ([WinError 10053]), aparentemente
    interferencia de software de seguridad local con la conexión loopback
    entre procesos. Esto no tiene relación con el entrenamiento en sí (CPU/GPU
    siguen funcionando con normalidad), pero si no se captura, la excepción
    se propaga y aborta TODO el entrenamiento, perdiendo horas de cómputo por
    un problema puramente de observabilidad. Se captura de forma amplia
    (`Exception`) a propósito: no vale la pena perder un entrenamiento de 30
    épocas por un fallo de sincronización de métricas que no afecta el
    resultado del modelo.
    """
    try:
        ejecucion.log(metricas)
    except Exception as error:  # noqa: BLE001 - fallo de red local, no debe abortar el entrenamiento
        print(f"[wandb] no se pudo registrar la época (se continúa el entrenamiento): {error}")


def guardar_historial_entrenamiento(historial_epocas: list[dict]) -> None:
    """Persiste el historial de épocas (loss/accuracy train y val) en un JSON
    local para que la app de Streamlit grafique las curvas de entrenamiento
    sin necesidad de credenciales ni conexión a W&B.

    Se regenera cada vez que se corre `train.py`, así que si se reentrena el
    modelo el archivo queda automáticamente sincronizado con el nuevo run.
    """
    RUTA_HISTORIAL_ENTRENAMIENTO.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_HISTORIAL_ENTRENAMIENTO, "w", encoding="utf-8") as archivo:
        json.dump(historial_epocas, archivo, indent=2)


def entrenar_modelo_completo() -> None:
    """Entrena ResNet18 desde cero en un único loop y registra todo en W&B.

    Al no haber pesos preentrenados que proteger, no hay fases: todos los
    parámetros del modelo se optimizan juntos desde la época 1, con early
    stopping monitoreando `val_loss` durante todo el entrenamiento y
    guardando siempre el mejor checkpoint visto.
    """
    fijar_semillas(CONFIGURACION["semilla"])
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de entrenamiento: {dispositivo}")

    cargador_train, cargador_val, _cargador_test = crear_dataloaders(
        DIRECTORIO_DATOS,
        tamano_lote=CONFIGURACION["tamano_lote"],
        proporcion_validacion=CONFIGURACION["proporcion_validacion"],
        semilla=CONFIGURACION["semilla"],
    )

    modelo = crear_modelo(probabilidad_dropout=CONFIGURACION["probabilidad_dropout"]).to(dispositivo)

    # Se evita el `with wandb.init(...) as ejecucion:` porque, en esta máquina,
    # un fallo de conectividad del servicio local de W&B durante el cierre
    # (`__exit__`) también puede propagar una excepción y hacer perder el
    # resultado del entrenamiento aunque el modelo ya se haya entrenado y
    # guardado correctamente. Se maneja el ciclo de vida manualmente para
    # poder capturar ese fallo de forma acotada en el `finally`.
    if wandb is not None:
        ejecucion = wandb.init(project=NOMBRE_PROYECTO_WANDB, config=CONFIGURACION)
    else:
        print("[wandb] paquete no instalado: el entrenamiento continúa sin registrar métricas en W&B.")
        ejecucion = _EjecucionSinWandb()
    try:
        entrenables, total_parametros = contar_parametros_entrenables(modelo)
        print(f"Parámetros entrenables: {entrenables:,} de {total_parametros:,} totales (100%, sin capas congeladas)")

        criterio = nn.BCEWithLogitsLoss()
        optimizador = AdamW(
            modelo.parameters(),
            lr=CONFIGURACION["tasa_aprendizaje"],
            weight_decay=CONFIGURACION["weight_decay"],
        )
        programador_lr = ReduceLROnPlateau(optimizador, mode="min", factor=0.5, patience=2)

        mejor_val_loss = float("inf")
        epocas_sin_mejora = 0
        historial_epocas: list[dict] = []

        for epoca in range(1, CONFIGURACION["epocas_maximas"] + 1):
            perdida_train, exactitud_train = entrenar_una_epoca(
                modelo, cargador_train, criterio, optimizador, dispositivo
            )
            perdida_val, exactitud_val = evaluar(modelo, cargador_val, criterio, dispositivo)
            programador_lr.step(perdida_val)

            print(
                f"época {epoca}/{CONFIGURACION['epocas_maximas']} - "
                f"train_loss={perdida_train:.4f} train_acc={exactitud_train:.4f} "
                f"val_loss={perdida_val:.4f} val_acc={exactitud_val:.4f}"
            )
            registrar_metricas_en_wandb(
                ejecucion,
                {
                    "epoca": epoca,
                    "train_loss": perdida_train,
                    "train_accuracy": exactitud_train,
                    "val_loss": perdida_val,
                    "val_accuracy": exactitud_val,
                    "tasa_aprendizaje_actual": optimizador.param_groups[0]["lr"],
                },
            )
            historial_epocas.append(
                {
                    "epoca": epoca,
                    "train_accuracy": exactitud_train,
                    "val_accuracy": exactitud_val,
                    "train_loss": perdida_train,
                    "val_loss": perdida_val,
                }
            )

            if perdida_val < mejor_val_loss:
                mejor_val_loss = perdida_val
                epocas_sin_mejora = 0
                RUTA_MEJOR_MODELO.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict_modelo": modelo.state_dict(),
                        "epoca_global": epoca,
                        "val_loss": perdida_val,
                        "val_accuracy": exactitud_val,
                        "fase": "entrenamiento_desde_cero",
                    },
                    RUTA_MEJOR_MODELO,
                )
                print(f"  -> nuevo mejor modelo guardado (val_loss={perdida_val:.4f})")
            else:
                epocas_sin_mejora += 1
                if epocas_sin_mejora >= CONFIGURACION["paciencia_early_stopping"]:
                    print(
                        f"  -> early stopping: {CONFIGURACION['paciencia_early_stopping']} "
                        "épocas sin mejorar val_loss"
                    )
                    break

        try:
            ejecucion.summary["mejor_val_loss"] = mejor_val_loss
        except Exception as error:  # noqa: BLE001 - ver registrar_metricas_en_wandb
            print(f"[wandb] no se pudo actualizar el resumen final: {error}")

        guardar_historial_entrenamiento(historial_epocas)

        print(f"Entrenamiento terminado. Mejor val_loss: {mejor_val_loss:.4f}")
        print(f"Checkpoint guardado en: {RUTA_MEJOR_MODELO}")
        print(f"Historial de entrenamiento guardado en: {RUTA_HISTORIAL_ENTRENAMIENTO}")
    finally:
        try:
            ejecucion.finish()
        except Exception as error:  # noqa: BLE001 - ver registrar_metricas_en_wandb
            print(
                "[wandb] no se pudo cerrar limpiamente la corrida "
                f"(el checkpoint entrenado no se ve afectado): {error}"
            )


if __name__ == "__main__":
    entrenar_modelo_completo()
