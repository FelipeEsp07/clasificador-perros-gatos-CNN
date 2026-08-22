"""
Entrenamiento del clasificador perro/gato con ResNet18 **desde cero**, sin
pesos preentrenados de ImageNet (sin transfer learning).

Este proyecto es un experimento comparativo explícito: `clasificador-perros-
gatos` usa EfficientNet-B0 con transfer learning (97.33% accuracy en test);
esta variante entrena la arquitectura ResNet18 completa desde inicialización
aleatoria, para medir cuánto aporta realmente el transfer learning frente a
aprender todo desde los ~8,700 ejemplos de entrenamiento disponibles.

Al no haber backbone preentrenado que proteger, el esquema de dos fases
(cabeza congelada → fine-tuning) de la versión anterior de este proyecto
(y de `clasificador-perros-gatos`) no aplica: aquí todas las capas se
entrenan juntas desde la época 1, con un único optimizador. Se compensa con:
- Un learning rate inicial más alto (1e-3 en vez de 1e-5 de fine-tuning),
  típico de entrenar una red convolucional desde cero.
- Muchas más épocas máximas (hasta 30, contra 22 de la variante con
  transfer learning), porque converger desde pesos aleatorios es
  sustancialmente más lento que ajustar un modelo ya preentrenado.
- Early stopping con paciencia algo mayor (6 épocas) para no cortar el
  entrenamiento prematuramente durante ese proceso de convergencia más lento.

El resto del pipeline (dataset, augmentation, criterio de pérdida,
optimizador AdamW, scheduler, checkpointing del mejor modelo por `val_loss`,
semillas) se mantiene igual al resto de proyectos de esta comparación.
"""

from __future__ import annotations

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

# Se registra el proyecto de W&B `clasificador-perros-gatos` (el mismo que
# usan las variantes con EfficientNet-B0 y ResNet18+transfer learning), a
# propósito: así los tres runs quedan en el mismo proyecto y se pueden
# comparar directamente desde el dashboard, filtrando por los campos
# `backbone` y `pretrained` del config.
NOMBRE_PROYECTO_WANDB = "clasificador-perros-gatos"

CONFIGURACION = {
    "arquitectura": "resnet18",
    "backbone": "resnet18",
    # Campo explícito para diferenciar este run (entrenamiento desde cero)
    # del run de ResNet18 con transfer learning y del de EfficientNet-B0.
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


def entrenar_modelo_completo() -> None:
    """Entrena ResNet18 desde cero en un único loop y registra todo en W&B.

    A diferencia de la variante con transfer learning, aquí no hay fases:
    todos los parámetros del modelo se optimizan juntos desde la época 1,
    con early stopping monitoreando `val_loss` sobre todo el entrenamiento
    (no por fase) y guardando siempre el mejor checkpoint visto.
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

        print(f"Entrenamiento terminado. Mejor val_loss: {mejor_val_loss:.4f}")
        print(f"Checkpoint guardado en: {RUTA_MEJOR_MODELO}")
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
