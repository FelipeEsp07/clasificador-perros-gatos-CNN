"""
Definición del modelo: arquitectura ResNet18 (sin pesos preentrenados) con
una cabeza de clasificación binaria (perro vs. gato) para entrenamiento
completo "desde cero".

El objetivo es entrenar ResNet18 desde cero, sin aprovechar pesos
preentrenados en ImageNet, para estudiar cómo se comporta la red cuando debe
aprender todas sus representaciones directamente de este dataset. Por eso:

1. Se crea `resnet18(weights=None)`: los pesos se inicializan al azar
   (inicialización estándar de PyTorch para capas convolucionales/lineales),
   no se descarga ni se usa ningún checkpoint de ImageNet.
2. Se reemplaza la cabeza de clasificación de 1000 clases (`modelo.fc`) por
   una cabeza binaria: `Dropout` (regularización) + `Linear` a 1 solo logit.
3. TODAS las capas quedan entrenables desde el inicio. No hay fases de
   "backbone congelado" ni fine-tuning progresivo: sin pesos preentrenados
   que preservar, ese esquema de dos fases no tiene sentido — el modelo
   completo debe aprender tanto las características de bajo nivel (bordes,
   texturas) como las de alto nivel (formas de perros/gatos) desde los datos
   de este dataset, por lo que se entrena todo junto con un único
   optimizador (ver `train.py`).

La salida sigue siendo un único logit (sin sigmoide aplicada dentro del
modelo) porque se usa `nn.BCEWithLogitsLoss`, más estable numéricamente que
aplicar `Sigmoid` + `BCELoss` por separado.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


def crear_modelo(probabilidad_dropout: float = 0.3) -> nn.Module:
    """Crea ResNet18 sin pesos preentrenados, con la cabeza binaria de salida.

    `weights=None` fuerza una inicialización aleatoria de todos los pesos
    (equivalente a `pretrained=False` en la API antigua de torchvision):
    el modelo no tiene ningún conocimiento previo de ImageNet, así que debe
    aprender absolutamente todo a partir del dataset de perros y gatos.
    Todos los parámetros quedan con `requires_grad=True` por defecto, sin
    necesidad de congelar/descongelar nada.
    """
    modelo = resnet18(weights=None)

    # En ResNet18 la capa final se llama `fc` y tiene 512 características
    # de entrada.
    numero_caracteristicas_entrada = modelo.fc.in_features
    modelo.fc = nn.Sequential(
        nn.Dropout(p=probabilidad_dropout, inplace=True),
        nn.Linear(numero_caracteristicas_entrada, 1),
    )

    return modelo


def contar_parametros_entrenables(modelo: nn.Module) -> tuple[int, int]:
    """Devuelve (parámetros entrenables, parámetros totales) del modelo.

    En este proyecto (entrenamiento desde cero) ambos valores deberían ser
    iguales, porque no se congela ninguna capa: se conserva esta función
    igualmente porque `train.py` la usa para reportar el tamaño del modelo
    de forma explícita al iniciar el entrenamiento.
    """
    total = sum(parametro.numel() for parametro in modelo.parameters())
    entrenables = sum(
        parametro.numel() for parametro in modelo.parameters() if parametro.requires_grad
    )
    return entrenables, total


if __name__ == "__main__":
    # Prueba de humo: verifica que el modelo produce la forma de salida
    # correcta (un logit por imagen) y que, al no congelar ninguna capa,
    # el 100% de los parámetros queda entrenable (resultado exacto y
    # verificable sin necesidad de entrenar nada).
    modelo = crear_modelo()

    lote_de_prueba = torch.randn(4, 3, 224, 224)  # 4 imágenes RGB de 224x224
    salida = modelo(lote_de_prueba)
    assert salida.shape == (4, 1), f"Se esperaba forma (4, 1), dio {tuple(salida.shape)}"

    entrenables, total = contar_parametros_entrenables(modelo)
    assert entrenables == total, (
        f"Al entrenar desde cero, todos los parámetros deben ser entrenables: "
        f"{entrenables:,} entrenables de {total:,} totales"
    )
    # ResNet18 tiene ~11.18M de parámetros totales (fijo por la arquitectura,
    # independientemente de si los pesos son aleatorios o preentrenados).
    assert total == 11_177_025, f"Se esperaban 11,177,025 parámetros totales, dio {total:,}"

    print(f"Forma de salida del modelo: {tuple(salida.shape)}")
    print(f"Parámetros totales del modelo: {total:,}")
    print(f"Parámetros entrenables (todos, sin congelar nada): {entrenables:,}")
