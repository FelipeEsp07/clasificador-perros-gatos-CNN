"""
Evaluación final del modelo entrenado sobre el `test_set`, el único
subconjunto que nunca se usó para tomar decisiones de diseño ni de
hiperparámetros durante el entrenamiento.

Reporta accuracy, precision, recall, F1 y la matriz de confusión completa
(no solo accuracy), porque en un problema binario balanceado la accuracy sola
puede esconder un desbalance entre falsos positivos y falsos negativos.

La lógica de evaluación no depende de la arquitectura del backbone, solo de
la interfaz que expone `model.py` (`crear_modelo` devolviendo un logit).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from tqdm import tqdm

from dataset import NOMBRES_CLASES, crear_dataloaders
from model import crear_modelo

DIRECTORIO_PROYECTO = Path(__file__).resolve().parent.parent
DIRECTORIO_DATOS = DIRECTORIO_PROYECTO / "data"
RUTA_MEJOR_MODELO = DIRECTORIO_PROYECTO / "models" / "best_model.pth"
RUTA_MATRIZ_CONFUSION = DIRECTORIO_PROYECTO / "models" / "matriz_confusion.png"


@torch.no_grad()
def predecir_sobre_test(
    modelo: nn.Module, cargador_test, dispositivo: torch.device
) -> tuple[list[int], list[int]]:
    """Recorre el test set y devuelve (etiquetas_reales, etiquetas_predichas)."""
    modelo.eval()
    etiquetas_reales: list[int] = []
    etiquetas_predichas: list[int] = []

    for imagenes, etiquetas in tqdm(cargador_test, desc="evaluación final"):
        imagenes = imagenes.to(dispositivo, non_blocking=True)
        logits = modelo(imagenes)
        predicciones = (torch.sigmoid(logits) >= 0.5).long().squeeze(1).cpu().tolist()

        etiquetas_reales.extend(int(etiqueta) for etiqueta in etiquetas.tolist())
        etiquetas_predichas.extend(predicciones)

    return etiquetas_reales, etiquetas_predichas


def calcular_metricas(etiquetas_reales: list[int], etiquetas_predichas: list[int]) -> dict:
    """Calcula accuracy, precision, recall y F1 (positivo = clase 'perro', etiqueta 1)."""
    return {
        "accuracy": accuracy_score(etiquetas_reales, etiquetas_predichas),
        "precision": precision_score(etiquetas_reales, etiquetas_predichas),
        "recall": recall_score(etiquetas_reales, etiquetas_predichas),
        "f1": f1_score(etiquetas_reales, etiquetas_predichas),
    }


def graficar_matriz_confusion(etiquetas_reales: list[int], etiquetas_predichas: list[int]) -> None:
    """Genera y guarda la matriz de confusión como imagen para inspección visual."""
    matriz = confusion_matrix(etiquetas_reales, etiquetas_predichas)
    figura, ejes = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=NOMBRES_CLASES,
        yticklabels=NOMBRES_CLASES,
        ax=ejes,
    )
    ejes.set_xlabel("Predicción")
    ejes.set_ylabel("Real")
    ejes.set_title("Matriz de confusión — test set")
    figura.tight_layout()
    RUTA_MATRIZ_CONFUSION.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(RUTA_MATRIZ_CONFUSION)
    plt.close(figura)


def evaluar_modelo_entrenado() -> dict:
    """Carga el mejor checkpoint y calcula las métricas finales sobre el test set."""
    if not RUTA_MEJOR_MODELO.exists():
        raise FileNotFoundError(
            f"No se encontró el checkpoint '{RUTA_MEJOR_MODELO}'. Ejecuta train.py primero."
        )

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = crear_modelo().to(dispositivo)
    estado_guardado = torch.load(RUTA_MEJOR_MODELO, map_location=dispositivo)
    modelo.load_state_dict(estado_guardado["state_dict_modelo"])
    print(
        f"Checkpoint cargado (época {estado_guardado['epoca_global']}, "
        f"fase '{estado_guardado['fase']}', val_loss={estado_guardado['val_loss']:.4f})"
    )

    _cargador_train, _cargador_val, cargador_test = crear_dataloaders(DIRECTORIO_DATOS)

    etiquetas_reales, etiquetas_predichas = predecir_sobre_test(modelo, cargador_test, dispositivo)
    metricas = calcular_metricas(etiquetas_reales, etiquetas_predichas)
    graficar_matriz_confusion(etiquetas_reales, etiquetas_predichas)

    return metricas


if __name__ == "__main__":
    metricas = evaluar_modelo_entrenado()

    print("\nMétricas finales sobre el test set:")
    print(f"  accuracy:  {metricas['accuracy']:.4f}")
    print(f"  precision: {metricas['precision']:.4f}")
    print(f"  recall:    {metricas['recall']:.4f}")
    print(f"  f1:        {metricas['f1']:.4f}")
    print(f"\nMatriz de confusión guardada en: {RUTA_MATRIZ_CONFUSION}")

    # No hay una assert de valor exacto esperado aquí porque el desempeño de
    # un modelo entrenado depende de la inicialización aleatoria y del
    # hardware; en su lugar, se valida el objetivo del proyecto (>90% de
    # accuracy en test) como condición mínima de éxito.
    assert metricas["accuracy"] > 0.90, (
        f"Se esperaba accuracy > 0.90 en el test set, dio {metricas['accuracy']:.4f}"
    )
